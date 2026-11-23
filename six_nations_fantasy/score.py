"""Fantasy scoring rules and scoring of observed performances."""

from typing import Callable

import numpy as np
import pandas as pd


# Each callable returns the fantasy points earned from one observed component.
SCORE_COMPONENTS = {
    "tries": lambda rows: 15 * rows["tries"],
    "try_assists": lambda rows: 4 * rows["try_assists"],
    "conversions": lambda rows: 2 * rows["conversions"],
    "penalty_goals": lambda rows: 3 * rows["penalty_goals"],
    "drop_goals": lambda rows: 5 * rows["drop_goals"],
    "defenders_beaten": lambda rows: 2 * rows["defenders_beaten"],
    "carried_meters": lambda rows: rows["carried_meters"] // 10,
    "completed_tackles": lambda rows: rows["completed_tackles"],
    "turnovers_won": lambda rows: 5 * rows["turnovers_won"],
    "offloads": lambda rows: 2 * rows["offloads"],
    "yellow_cards": lambda rows: -5 * rows["yellow_cards"],
    "red_cards": lambda rows: -8 * rows["red_cards"],
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
