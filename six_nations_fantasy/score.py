"""Fantasy scoring rules and scoring of observed performances."""

from typing import Callable

import numpy as np
import pandas as pd


# TODO should try to move away from per-row operations and focus on 
# working with the columns directly (maybe)?
def score(player, try_points=15): 
    return try_points * player['tries'] \
        + 4 * player['try_assists'] \
        + 2 * player['conversions'] \
        + 3 * player['penalty_goals'] \
        + 5 * player['drop_goals'] \
        + 2 * player['defenders_beaten'] \
        + player['carried_meters'] // 10 \
        + player['completed_tackles'] \
        + 5 * player['turnovers_won'] \
        + 2 * player['offloads'] \
        - 5 * player['yellow_cards'] \
        - 8 * player['red_cards']

def score_back(player):
    return score(score, try_points=10)

def score_forward(player):
    return score(player)


def score_performances(
    performances: pd.DataFrame, scorer: Callable[[pd.Series], float]
) -> pd.Series:
    """Score completed performances without modifying the underlying table.

    This helper is used by the evaluator for actual outcomes and can be used
    by models that learn from historical points. Other models need not call it.
    """
    try:
        values = performances.apply(scorer, axis=1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not calculate fantasy points from performances: {exc}") from exc
    try:
        result = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scorer must return numerical fantasy points") from exc
    if not np.isfinite(result).all():
        raise ValueError("Scorer returned missing or non-finite fantasy points")
    return pd.Series(result, index=performances.index, name="fantasy_points")
