"""Discover completed senior men's tests involving a Six Nations side."""

from collections.abc import Mapping
from datetime import date, datetime
from itertools import chain
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import pandas as pd
import requests

from .source import embedded_json, match_id

TEAMS = ("England", "France", "Ireland", "Italy", "Scotland", "Wales")
COMPETITIONS = {
    "six-nations": (0, "Six Nations"),
    "internationals": (5, "Internationals"),
    "rugby-world-cup": (6, "Rugby World Cup"),
    "autumn-nations-series": (7, "Autumn Nations Series"),
    "nations-championship": (8, "Nations Championship"),
}
NON_TEST_OPPONENTS = {
    "all blacks xv", "maori all blacks", "māori all blacks",
    "barbarians", "barbarian fc", "barbarians fc", "british & irish lions",
    "british and irish lions", "french barbarians", "new zealand maori",
}
BASE_URL = "https://www.rugbypass.com"
MANIFEST_COLUMNS = [
    "match_id", "url", "date", "home_team", "away_team", "home_score",
    "away_score", "competition_id", "competition", "competition_slug",
    "year", "season", "season_id",
]


def fixture_key(row):
    """Identify the same fixture across URL changes and home/away display order."""
    return (pd.Timestamp(row["date"]).date(),
            *sorted((row["home_team"], row["away_team"])))


def _is_test(slug, home, away):
    return (
        slug in COMPETITIONS
        and (home in TEAMS or away in TEAMS)
        and all(name.casefold() not in NON_TEST_OPPONENTS
                and not name.casefold().endswith((" xv", " a", " women", " u20", " under 20"))
                for name in (home, away))
    )


def _items(value):
    return value.values() if isinstance(value, Mapping) else value


def _structured_results(payload, year):
    for day in chain(*(_items(payload[key]) for key in ("currentGameDays", "currentFinalGameDays"))):
        for tournament in _items(day["tournaments"]):
            for game in _items(tournament["games"]):
                if not game.get("played") or not game.get("hasLink"):
                    continue
                slug = game["tournamentUri"]
                home, away = (game[side + "Team"]["name"] for side in ("home", "away"))
                if _is_test(slug, home, away):
                    if match_id(game["url"]) != int(game["id"]):
                        raise ValueError(f"Invalid listing URL for fixture {game['id']}")
                    yield _fixture(
                        game["url"], datetime.strptime(game["dateId"], "%Y%m%d").date(),
                        slug, home, away, int(game["homeScore"]), int(game["awayScore"]), year * 100,
                    )


def _fixture(url, day, slug, home, away, home_score, away_score, season_id):
    competition_id, competition = COMPETITIONS[slug]
    return dict(match_id=match_id(url), url=url, date=day, home_team=home, away_team=away,
                home_score=home_score, away_score=away_score, competition_id=competition_id,
                competition=competition, competition_slug=slug, year=day.year,
                season=str(day.year), season_id=season_id)


def _team_results(html):
    soup = BeautifulSoup(html, "html.parser")
    for day in soup.select(".date-section"):
        match_date = datetime.strptime(day.select_one(".date").get_text(strip=True),
                                       "%a %d %b, %Y").date()
        for tournament in day.select(".comp-section"):
            title = tournament.select_one(".title a[href]")
            if title is None:
                continue
            slug = urlsplit(title["href"]).path.strip("/")
            for game in tournament.select(".comp-game"):
                link = game.select_one("a.link-box[href]")
                if not game.select_one(".full-time") or link is None:
                    continue
                home, away = (game.select_one(f".team.{side}").get_text(strip=True)
                              for side in ("home", "away"))
                if _is_test(slug, home, away):
                    hs, aws = (int(score.get_text(strip=True)) for score in game.select(".score"))
                    yield _fixture(link["href"], match_date, slug, home, away, hs, aws,
                                   match_date.year * 100)


def _score(row):
    return dict(zip((row["home_team"], row["away_team"]),
                    (row["home_score"], row["away_score"])))


def _resolve_team_result(session, listing):
    response = session.get(listing["url"], timeout=30)
    response.raise_for_status()
    _, _, state = response.text.partition("const mcApp")
    games = [embedded_json(state, "game"), *embedded_json(state, "lastFiveGames")]
    for game in games:
        slug = game["compUri"]
        home, away = (game[side + "Team"]["name"] for side in ("home", "away"))
        if not _is_test(slug, home, away):
            continue
        game_row = _fixture(
            f"{game['baseUrl']}?g={game['id']}", date.fromisoformat(game["date"][:10]),
            slug, home, away, int(game["homeTeam"]["score"]), int(game["awayTeam"]["score"]),
            int(game["season"]),
        )
        if fixture_key(game_row) == fixture_key(listing) and _score(game_row) == _score(listing):
            return game_row
    raise ValueError("No match-centre record agrees with the calendar date, opponents and score")


def _post(session, path, **form):
    response = session.post(f"{BASE_URL}/{path}/",
                            files={key: (None, str(value)) for key, value in form.items()},
                            timeout=30)
    response.raise_for_status()
    return response.json()


def discover_international_urls(years, existing_matches):
    """Return pending fixture URLs and report team listings with no verified URL."""
    existing_ids = set(existing_matches["match_id"])
    existing_keys = {fixture_key(match) for match in existing_matches.to_dict("records")}
    selected, unresolved, attempted = {}, [], set()
    with requests.Session() as session:
        for year in years:
            print(f"Discovering international fixtures for {year}...", flush=True)
            aggregate = _post(session, "internationals/fixtures-results", action="filter-fixtures",
                              season=year * 100, team=0, comp=0, isContent=1)
            for row in _structured_results(aggregate, year):
                selected[fixture_key(row)] = row
            for team in TEAMS:
                calendar = _post(session, f"teams/{team.lower()}", action="get-fixtures",
                                 season=year, isContent=1)
                for row in _team_results(calendar["fixtures"]):
                    key = fixture_key(row)
                    attempt = (key, row["url"])
                    if key in existing_keys or key in selected or attempt in attempted:
                        continue
                    attempted.add(attempt)
                    try:
                        selected[key] = _resolve_team_result(session, row)
                    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
                        unresolved.append({**row, "error": str(exc)})

    pending = [row for key, row in selected.items()
               if key not in existing_keys and row["match_id"] not in existing_ids]
    failures = pd.DataFrame(
        [row for row in unresolved if fixture_key(row) not in selected],
        columns=[*MANIFEST_COLUMNS, "error"],
    ).drop_duplicates(["date", "match_id"])
    return (
        pd.DataFrame(pending, columns=MANIFEST_COLUMNS)
        .set_index("match_id", verify_integrity=True).sort_index(),
        failures,
    )
