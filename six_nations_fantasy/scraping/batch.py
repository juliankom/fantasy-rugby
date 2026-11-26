"""Scrape a URL manifest into a separate dataset and record rejected fixtures."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from .match import scrape_match


def scrape_urls(urls, directory):
    """Publish a new directory with successful matches and one row per failure."""
    target = Path(directory)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if urls.index.name != "match_id" or not urls.index.is_unique:
        raise ValueError("Expected a manifest indexed by unique match_id")
    matches, teams, players, failures = [], [], [], []
    with requests.Session() as session:
        for key, fixture in urls.iterrows():
            try:
                result = scrape_match(fixture.url, session=session)
                if key != result.match["match_id"]:
                    raise ValueError("Manifest and match ID disagree")
                for field in ("competition_slug", "season_id", "home_team", "away_team",
                              "home_score", "away_score"):
                    if fixture[field] != result.match[field]:
                        raise ValueError(f"Manifest and match disagree on {field}")
                if "date" in fixture and pd.Timestamp(fixture["date"]).date() != result.match["date"]:
                    raise ValueError("Manifest and match disagree on date")
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                failures.append(dict(match_id=key, url=fixture.url, error=str(exc)))
                continue
            matches.append({**fixture.to_dict(), **result.match})
            teams.extend(result.teams)
            players.extend(result.performances)
    errors = pd.DataFrame(failures, columns=["match_id", "url", "error"])
    with TemporaryDirectory(dir=target.parent) as tmp:
        output = Path(tmp) / "dataset"
        output.mkdir()
        errors.to_parquet(output / "failures.parquet", index=False)
        if matches:
            perfs = pd.DataFrame(players)
            registry = perfs[["player_id", "player_identifier", "name"]].drop_duplicates(
                "player_id", keep="last").set_index("player_id")
            registry.to_parquet(output / "players.parquet")
            perfs.drop(columns=["player_identifier", "name"]).rename_axis("perf_id").to_parquet(
                output / "player_performances.parquet")
            pd.DataFrame(matches).set_index("match_id").to_parquet(output / "matches.parquet")
            pd.DataFrame(teams).to_parquet(output / "team_performances.parquet", index=False)
        output.rename(target)
    return errors
