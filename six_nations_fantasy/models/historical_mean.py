"""Historical mean fantasy-points model."""

from typing import Callable

import pandas as pd

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.score import score_performances


class HistoricalMean:
    """Mean score by player, using global training mean for unseen players.

    This model needs fantasy points during fitting and calculates
    them from the observed historical performances with the supplied scorer.
    """

    def __init__(self, *, scorer: Callable[[pd.Series], float]) -> None:
        self.scorer = scorer

    def fit(self, history: RugbyDataset) -> None:
        if history.performances.empty:
            raise ValueError("Cannot fit on empty history")
        performances = history.performances
        scores = score_performances(performances, self.scorer)
        # Group by column values (not the row index), keeping all player rows.
        self.global_mean_ = float(scores.mean())
        self.player_means_ = scores.groupby(performances["player_id"]).mean()

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "player_means_"):
            raise RuntimeError("Call fit() before predict()")
        return pd.Series(
            fixtures["player_id"].map(self.player_means_).fillna(self.global_mean_).to_numpy(),
            index=fixtures.index,
            name="predicted",
        )
