"""Parse a completed RugbyPass match from one HTTP response."""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit

import requests

from .source import embedded_json, match_id

PLAYER_STATS = (
    "carries", "clean_breaks", "completed_tackles", "turnovers_conceded",
    "turnovers_won", "total_kicks", "passes", "carries_metres", "offloads",
    "defenders_beaten", "try_assists", "tries", "carries_per_minute",
    "total_tackles", "missed_tackles", "dominant_tackles", "ruck_turnovers",
    "lineouts_won", "tackles_per_minute", "red_cards", "yellow_cards",
    "penalties_conceded",
)


@dataclass
class MatchResult:
    match: dict
    teams: list[dict]
    performances: list[dict]


def minutes_played(number, changes):
    """Sum stints on the pitch, using the notebook's 80-minute cap."""
    start = 0 if number <= 15 else None
    total = 0
    for event in sorted(changes, key=lambda e: (int(e["minute"]), int(e["second"]))):
        minute = min(int(event["minute"]), 80)
        if event["type"] == "sub-off" and start is not None:
            total += minute - start
            start = None
        elif event["type"] == "sub-on" and start is None:
            start = minute
        else:
            raise ValueError(f"Inconsistent substitution for shirt {number}: {event}")
    return total if start is None else total + 80 - start


def _performances(game, stats, game_id):
    rows = {}
    for side in ("home", "away"):
        lineup = game[side + "Players"]
        if not lineup:
            raise ValueError(f"Missing {side} team sheet")
        for player in lineup:
            slug = player["uri"]
            key = (side == "home", slug)
            if not slug or key in rows:
                raise ValueError(f"Missing or duplicate lineup identifier: {key}")
            number = int(player["pos_id"])
            events = Counter(e["type"] for e in player["events"])
            rows[key] = dict(
                match_id=game_id, home=key[0], number=number, name=player["fname"],
                player_identifier=slug, player_id=int(player["id"]),
                minutes_played=minutes_played(number, player["subEvents"]),
                conversions=events["conversion"], penalty_goals=events["penalty"],
                drop_goals=events["drop_goal"] + events["drop-goal"],
            )
    if not stats["carries"] or not stats["completed_tackles"]:
        raise ValueError("Missing player activity in completed match")
    for stat in PLAYER_STATS:
        column = "carried_meters" if stat == "carries_metres" else stat
        # RugbyPass lists only the players credited with this stat.
        for row in rows.values():
            row[column] = 0
        seen = set()
        for record in stats[stat]:
            slug = urlsplit(record["link"]).path.rstrip("/").split("/")[-1]
            key = (record["home"], slug)
            if key not in rows or key in seen:
                raise ValueError(f"Unmatched or duplicate {stat} record: {key}")
            seen.add(key)
            cast = float if stat.endswith("_per_minute") else int
            rows[key][column] = cast(record["total"])
    return list(rows.values())


def _value(text):
    text = str(text)
    if text.endswith("%"):
        return float(text[:-1]) / 100
    if ":" in text:
        kicks, passes = map(float, text.split(":"))
        return kicks / (kicks + passes) if kicks + passes else None
    return int(text.removesuffix("m"))


def _team_rows(stats, game_id):
    rows = []
    for side in ("home", "away"):
        row = {"match_id": game_id, "home": side == "home"}
        for section in ("statsSummary", "setPlays", "attack", "turnovers",
                        "penalties", "defence", "kicks"):
            for stat in stats[section]:
                row[stat["title"].lower().replace(" ", "_")] = _value(stat[side])
        possession = stats["possession"]
        row["possession_last_10_min"] = _value(possession[side + "Percent10"])
        territory = stats["territory"]["percents"]
        for i, value in enumerate(territory if side == "home" else reversed(territory)):
            row[f"territory_percents_{i}"] = _value(value)
        for i, value in enumerate(possession[side + "Data"]["percents"]):
            row[f"poss_percents_{i}"] = _value(value)
        rows.append(row)
    return rows


def parse_match(html, url):
    expected = match_id(url)
    _, marker, state = html.partition("const mcApp")
    if not marker:
        raise ValueError("Missing RugbyPass match centre")
    game = embedded_json(state, "game")
    if int(game["id"]) != expected:
        raise ValueError(f"Requested match {expected}, received {game['id']}")
    if game["dateStatus"] == "Full Time" and not game["hasTeamSheets"]:
        raise ValueError(f"Completed match {expected} has no team sheet")
    if not game["finished"]:
        raise ValueError(f"Match {expected} has no final result")
    stats = embedded_json(state, "stats")
    match = dict(match_id=expected, url=url, date=date.fromisoformat(game["date"][:10]),
                 competition_slug=game["compUri"], season_id=int(game["season"]),
                 extra_info=game["infoItems"][1]["title"].split(" - ", 1)[-1])
    for side in ("home", "away"):
        team = game[side + "Team"]
        match[side + "_team"] = team["name"]
        match[side + "_score"] = int(team["score"])
    teams = _team_rows(stats["teamStats"], expected)
    players = _performances(game, stats["playerStats"]["stats"], expected)
    for team in teams:
        for stat in ("conversions", "penalty_goals", "drop_goals"):
            credited = sum(p[stat] for p in players if p["home"] == team["home"])
            if credited != team[stat]:
                raise ValueError(f"Player and team {stat} disagree in match {expected}")
    return MatchResult(match, teams, players)


def scrape_match(url, *, session=None):
    response = (session or requests).get(url, timeout=30)
    response.raise_for_status()
    return parse_match(response.text, url)
