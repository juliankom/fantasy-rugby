"""Six Nations round schedules and prepared fantasy lineups."""

from pathlib import Path

import pandas as pd

from six_nations_fantasy.score import score_performances, simple_score


def fixture_path(directory, year):
    return Path(directory) / f"six_nations_{year}_fixtures.parquet"


def tournament_rounds(matches, year):
    season = matches.loc[matches["competition"].eq("Six Nations") & matches["date"].dt.year.eq(year)]
    weekends = season.groupby(season["date"].dt.to_period("W-SUN"), sort=True)
    if len(weekends) != 5:
        raise ValueError(f"Expected five Six Nations match weekends in {year}, found {len(weekends)}")
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
