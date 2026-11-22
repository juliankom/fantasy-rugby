"""Rolling round evaluation on announced rosters or recorded lineups."""

import pandas as pd

from six_nations_fantasy.data.fantasy_fixtures import fixture_path, tournament_rounds, with_actual_scores
from six_nations_fantasy.evaluation.pipeline import recorded_lineup
from six_nations_fantasy.score import score_performances, simple_score


def rolling_rounds(dataset, year, directory, *, require_prices):
    path = fixture_path(directory, year)
    roster = pd.read_parquet(path) if require_prices or path.exists() else None
    matches = dataset.matches
    for number, scheduled in tournament_rounds(matches, year):
        holdout = dataset.select_matches(scheduled["match_id"])
        if roster is None:
            fixtures = recorded_lineup(holdout)
            actual = score_performances(holdout.performances, simple_score)
            pool = None
        else:
            entries = roster.loc[roster["round"].eq(number)]
            if entries.empty:  # Historical workbooks may cover only some rounds.
                continue
            pool = with_actual_scores(entries, holdout)
            fixtures = pool[["match_id", "player_id", "date", "team", "shirt_number", "is_sub"]].rename(
                columns={"shirt_number": "number", "is_sub": "is_substitute"}
            )
            actual = pool["actual"]
        cutoff = scheduled["date"].min()
        history = dataset.select_matches(matches.loc[matches["date"] < cutoff, "match_id"])
        yield number, history, fixtures, actual.to_numpy(), pool


def summarize(rounds):
    """Pool squared player errors and compare total squad points across rounds."""
    rows = pd.DataFrame(rounds)
    totals = rows.groupby("model", sort=False)[["players", "squared_error", "squad_points", "optimum"]].sum(min_count=1)
    totals["player_RMSE"] = (totals.pop("squared_error") / totals["players"]) ** 0.5
    totals["% of optimum"] = 100 * totals["squad_points"] / totals["optimum"]
    return totals.dropna(axis="columns", how="all")
