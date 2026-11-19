"""Recency-weighted fantasy-points model."""

from typing import Callable

import pandas as pd

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.score import score_performances

from ._mean import MeanScores


class RecencyWeightedMean:
    """Exponentially weighted mean by player and lineup role when available."""

    def __init__(
        self, *, scorer: Callable[[pd.DataFrame], pd.Series], half_life_days: float = 180.0
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.scorer = scorer
        self.half_life_days = half_life_days

    def fit(self, history: RugbyDataset) -> None:
        if history.performances.empty:
            raise ValueError("Cannot fit on empty history")
        performances = history.performances
        scores = score_performances(performances, self.scorer)
        match_dates = history.matches.set_index("match_id")["date"]
        dates = performances["match_id"].map(match_dates)
        # Using the last date in training as reference avoids numerical overflow;
        # the common scaling cancels out in each weighted average.
        age = (dates.max() - dates).dt.days
        weight = 0.5 ** (age / self.half_life_days)
        self._scores = MeanScores(performances, scores, weight)

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "_scores"):
            raise RuntimeError("Call fit() before predict()")
        return self._scores.predict(fixtures)
