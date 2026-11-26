"""Six Nations round schedules and prepared fantasy lineups."""

from pathlib import Path
from warnings import warn

import pandas as pd

from six_nations_fantasy.score import score_performances, simple_score


def fixture_path(directory, year):
    return Path(directory) / f"six_nations_{year}_fixtures.parquet"


def tournament_rounds(matches, year, *, price_free=False):
    season = matches.loc[matches["competition"].eq("Six Nations") & matches["date"].dt.year.eq(year)]
    weekends = season.groupby(season["date"].dt.to_period("W-SUN"), sort=True)
    if len(weekends) != 5:
        warn(f"{year} Six Nations: found {len(weekends)} match weekends, expected 5. "
             "Only available fixtures can be evaluated.", stacklevel=2)
        if season.empty:
            return
        if "extra_info" in season:
            numbers = season["extra_info"].astype("string").str.extract(
                r"(?i)^\s*round\s+([1-5])\s*$", expand=False
            )
            if numbers.notna().all():
                yield from season.groupby(numbers.astype(int), sort=True)
                return
        if not price_free:
            raise ValueError(f"Cannot align the incomplete {year} schedule with roster round numbers")
        warn(f"{year}: round labels unavailable. Numbering observed weekends chronologically.",
             stacklevel=2)
    for round_number, (_, group) in enumerate(weekends, start=1):
        yield round_number, group


def with_actual_scores(roster, holdout):
    """Score recorded performances. An announced player who did not appear earns zero."""
    performances = holdout.performances
    actual = performances[["match_id", "player_id"]].assign(
        actual=score_performances(performances, simple_score)
    )
    scored_fixtures = roster.merge(actual, on=["match_id", "player_id"], how="left", validate="one_to_one")
    scored_fixtures["actual"] = scored_fixtures["actual"].fillna(0)
    return scored_fixtures
