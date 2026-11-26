"""Tournament evaluation on announced rosters or recorded lineups."""

import pandas as pd

from six_nations_fantasy.data.fantasy_fixtures import with_actual_scores
from six_nations_fantasy.data.lineup import match_context
from six_nations_fantasy.evaluation.pipeline import recorded_lineup
from six_nations_fantasy.ilp import POSITION_COUNTS
from six_nations_fantasy.models.position_shrinkage import STARTING_POSITIONS
from six_nations_fantasy.score import score_performances, simple_score


def scored_player_fixtures(round_matches, round_number, fantasy_roster):
    """Attach realized points to each announced player, or use the recorded lineup."""
    if fantasy_roster is None:
        lineup = recorded_lineup(round_matches)
        actual = score_performances(round_matches.performances, simple_score)
        return lineup.assign(actual=actual.to_numpy())

    entries = fantasy_roster.loc[fantasy_roster["round"].eq(round_number)]
    return match_context(with_actual_scores(entries, round_matches), round_matches.matches)


def prediction_lineup(scored_fixtures):
    """Pass only player identity and pre-match lineup information to predict()."""
    lineup = scored_fixtures.rename(columns={"shirt_number": "number", "is_sub": "is_substitute"})
    return lineup.filter(items=["match_id", "player_id", "date", "team", "opponent", "competition", "home", "number", "is_substitute"])


def summarize(rounds):
    """Pool squared player errors and compare total squad points across rounds."""
    rows = pd.DataFrame(rounds)
    totals = rows.groupby("model", sort=False)[["players", "squared_error", "squad_points", "optimum"]].sum(min_count=1)
    totals["player_RMSE"] = (totals.pop("squared_error") / totals["players"]) ** 0.5
    totals["% of optimum"] = 100 * totals["squad_points"] / totals["optimum"]
    return totals.dropna(axis="columns", how="all")


def price_free_picks(fixtures, predicted):
    """Rank starters within shirt groups and choose one replacement separately."""
    position = pd.to_numeric(fixtures["number"], errors="coerce").map(STARTING_POSITIONS)
    position = position.mask(fixtures["is_substitute"].fillna(False), "Bench")
    if position.isna().any():
        raise ValueError("Price-free selection requires a starting shirt or bench role for each player")
    ranked = fixtures.assign(position=position, predicted=predicted).sort_values(
        ["predicted", "player_id", "match_id"], ascending=[False, True, True]
    )
    slots = {**POSITION_COUNTS, "Bench": 1}
    picks = ranked.loc[ranked.groupby("position").cumcount() < ranked["position"].map(slots)]
    if len(picks) != sum(slots.values()):
        raise ValueError("Too few eligible players to fill the price-free positions")
    return picks


def compare_price_free_picks(picks, baseline):
    """Compare the same round's selections, counting each incoming player once."""
    result = {}
    for label, positions in [("XV", POSITION_COUNTS), ("bench", ["Bench"])]:
        group = picks.loc[picks["position"].isin(positions)]
        reference = baseline.loc[baseline["position"].isin(positions)]
        points = group["actual"].sum()
        result[f"{label} points"] = points
        result[f"{label} gain"] = points - reference["actual"].sum()
        result[f"{label} changes"] = int((~group["player_id"].isin(reference["player_id"])).sum())
    return result
