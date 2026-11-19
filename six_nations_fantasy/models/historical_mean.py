"""Historical mean fantasy-points model."""

from typing import Callable

import pandas as pd

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.score import score_performances

from ._mean import MeanScores


class HistoricalMean:
    """Mean score by player and lineup role when available.

    This model needs fantasy points during fitting and calculates
    them from the observed historical performances with the supplied scorer.
    """

    def __init__(self, *, scorer: Callable[[pd.DataFrame], pd.Series]) -> None:
        self.scorer = scorer

    def fit(self, history: RugbyDataset) -> None:
        if history.performances.empty:
            raise ValueError("Cannot fit on empty history")
        performances = history.performances
        scores = score_performances(performances, self.scorer)
        self._scores = MeanScores(performances, scores)

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "_scores"):
            raise RuntimeError("Call fit() before predict()")
        return self._scores.predict(fixtures)
