"""Optional conversion of observed player performances to fantasy points."""

from typing import Callable

import numpy as np
import pandas as pd


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
