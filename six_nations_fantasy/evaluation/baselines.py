"""Minimal baselines conforming to the evaluation interface."""

from typing import Callable

import pandas as pd

from six_nations_fantasy.data import RugbyDataset

from .scoring import score_performances


class HistoricalMean:
    """Mean score by player, using global training mean for unseen players.

    Only this baseline needs fantasy points during fitting; it calculates
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


class RecencyWeightedMean:
    """Exponentially weighted historical mean; half-life measured in days."""

    def __init__(
        self, *, scorer: Callable[[pd.Series], float], half_life_days: float = 180.0
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.scorer = scorer
        self.half_life_days = half_life_days

    def fit(self, history: RugbyDataset) -> None:
        if history.performances.empty:
            raise ValueError("Cannot fit on empty history")
        performances = history.performances
        scores = score_performances(performances, self.scorer).to_numpy()
        match_dates = history.matches.set_index("match_id")["date"]
        dates = performances["match_id"].map(match_dates)
        # Using the last date in training as reference avoids numerical overflow;
        # the common scaling cancels out in each weighted average.
        age = (dates.max() - dates).dt.days
        weight = 0.5 ** (age / self.half_life_days)
        x = pd.DataFrame({
            "player_id": performances["player_id"].to_numpy(),
            "_weighted": scores * weight.to_numpy(),
            "_weight": weight.to_numpy(),
        })
        sums = x.groupby("player_id")[["_weighted", "_weight"]].sum()
        self.player_means_ = sums["_weighted"] / sums["_weight"]
        self.global_mean_ = float(x["_weighted"].sum() / x["_weight"].sum())

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "player_means_"):
            raise RuntimeError("Call fit() before predict()")
        return pd.Series(
            fixtures["player_id"].map(self.player_means_).fillna(self.global_mean_).to_numpy(),
            index=fixtures.index,
            name="predicted",
        )
