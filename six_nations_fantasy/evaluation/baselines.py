"""Minimal baselines conforming to the evaluation interface."""

import pandas as pd

from .pipeline import TARGET_COLUMN


class HistoricalMean:
    """Mean score by player, using global training mean for unseen players."""

    def fit(self, history: pd.DataFrame) -> None:
        if history.empty:
            raise ValueError("Cannot fit on empty history")
        self.global_mean_ = float(history[TARGET_COLUMN].mean())
        self.player_means_ = history.groupby("player_id")[TARGET_COLUMN].mean()

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

    def __init__(self, *, half_life_days: float = 180.0) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = half_life_days

    def fit(self, history: pd.DataFrame) -> None:
        if history.empty:
            raise ValueError("Cannot fit on empty history")
        # Using the last date in training as reference avoids numerical overflow;
        # the common scaling cancels out in each weighted average.
        age = (history["date"].max() - history["date"]).dt.days
        weight = 0.5 ** (age / self.half_life_days)
        x = history.assign(_weighted=history[TARGET_COLUMN] * weight, _weight=weight)
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
