"""Fantasy scoring rules and scoring of observed performances."""

from typing import Callable

import numpy as np
import pandas as pd

def simple_score(performances: pd.DataFrame, try_points=15): 
    """Calculate fantasy scores from performance data.
    """
    return try_points * performances['tries'] \
        + 4 * performances['try_assists'] \
        + 2 * performances['conversions'] \
        + 3 * performances['penalty_goals'] \
        + 5 * performances['drop_goals'] \
        + 2 * performances['defenders_beaten'] \
        + performances['carried_meters'] // 10 \
        + performances['completed_tackles'] \
        + 5 * performances['turnovers_won'] \
        + 2 * performances['offloads'] \
        - 5 * performances['yellow_cards'] \
        - 8 * performances['red_cards']

def score_performances(
    performances: pd.DataFrame, scorer: Callable[[pd.DataFrame], pd.Series]
) -> pd.Series:
    """Score completed performances without modifying the underlying table.

    This helper is used by the evaluator for actual outcomes and can be used
    by models that learn from historical points. Other models need not call it.
    """
    try:
        values = scorer(performances)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not calculate fantasy points from performances: {exc}") from exc
    try:
        result = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scorer must return numerical fantasy points") from exc
    if not np.isfinite(result).all():
        raise ValueError("Scorer returned missing or non-finite fantasy points")
    return pd.Series(result, index=performances.index, name="fantasy_points")
