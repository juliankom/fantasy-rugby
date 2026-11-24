"""Fantasy scoring rules and scoring of observed performances."""

from typing import Callable

import numpy as np
import pandas as pd


# One scoring unit is one event, except metres which count in blocks of ten.
POINTS_PER_UNIT = {
    "tries": 15, "try_assists": 4, "conversions": 2, "penalty_goals": 3,
    "drop_goals": 5, "defenders_beaten": 2, "carried_meters": 1,
    "completed_tackles": 1, "turnovers_won": 5, "offloads": 2,
    "yellow_cards": -5, "red_cards": -8,
}


def scoring_units(rows, stat):
    return rows[stat] // 10 if stat == "carried_meters" else rows[stat]


def _component_scorer(stat, weight):
    return lambda rows: weight * scoring_units(rows, stat)


SCORE_COMPONENTS = {
    stat: _component_scorer(stat, weight) for stat, weight in POINTS_PER_UNIT.items()
}


def simple_score(performances: pd.DataFrame, try_points=15):
    """Sum the point contributions of an observed performance."""
    components = SCORE_COMPONENTS
    if try_points != 15:
        components = {**components, "tries": lambda rows: try_points * rows["tries"]}
    return sum(component(performances) for component in components.values())


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
