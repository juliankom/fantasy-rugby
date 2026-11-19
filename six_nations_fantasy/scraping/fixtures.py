"""Discover completed fixtures for the seasons covered by the original URL table."""

from itertools import chain

import pandas as pd
import requests

from .source import embedded_json, match_id

# competition_id, project name, RugbyPass path, season ending years
COMPETITIONS = (
    (0, "Six Nations", "six-nations", range(2022, 2027)),
    (1, "URC", "united-rugby-championship", range(2023, 2027)),
    (2, "Top 14", "top-14", range(2023, 2027)),
    (3, "Prem", "premiership", range(2023, 2027)),
    (4, "Champions Cup", "european-champions-cup", range(2023, 2027)),
)


def _fixtures(payload, season, slug):
    games = {}
    for day in chain(payload["currentGameDays"], payload["currentFinalGameDays"]):
        for tournament in day["tournaments"]:
            for game in tournament["games"]:
                key = int(game["id"])
                if game["tournamentUri"] != slug:
                    raise ValueError(f"Unexpected tournament for fixture {key}")
                if key in games and games[key] != game:
                    raise ValueError(f"Conflicting fixture {key}")
                games[key] = game
    if len(games) != int(season["games"]):
        raise ValueError(f"Incomplete listing for {slug} {season['label']}: "
                         f"{len(games)} distinct fixtures, expected {season['games']}")
    for key, game in games.items():
        if game["played"]:
            if not game["hasLink"] or match_id(game["url"]) != key:
                raise ValueError(f"Invalid URL for completed fixture {key}")
            yield dict(match_id=key, url=game["url"], home_team=game["homeTeam"]["name"],
                       away_team=game["awayTeam"]["name"], home_score=int(game["homeScore"]),
                       away_score=int(game["awayScore"]))


def discover_urls(competitions=COMPETITIONS):
    rows = []
    with requests.Session() as session:
        for competition_id, name, slug, years in competitions:
            endpoint = f"https://www.rugbypass.com/{slug}/fixtures-results/"
            page = session.get(endpoint, timeout=30)
            page.raise_for_status()
            seasons = {max(map(int, s["years"])): s for s in embedded_json(page.text, "seasons")}
            for year in years:
                season = seasons[year]
                form = dict(action="filter-fixtures", team="0", comp="0",
                            season=str(season["season"]), isContent="1")
                response = session.post(endpoint, files={k: (None, v) for k, v in form.items()},
                                        timeout=30)
                response.raise_for_status()
                for fixture in _fixtures(response.json(), season, slug):
                    rows.append(dict(**fixture, competition_id=competition_id, competition=name,
                                     competition_slug=slug, year=year, season=season["label"],
                                     season_id=season["season"]))
    return pd.DataFrame(rows).set_index("match_id", verify_integrity=True).sort_index()
