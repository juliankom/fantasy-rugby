"""Predict fantasy points from the lineup and a player's last five appearances."""

from typing import Callable

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.data.lineup import substitute_status
from six_nations_fantasy.score import score_performances


class RandomForestPoints:
    """Fit one forest across players, using one example per recorded appearance."""

    def __init__(self, *, scorer: Callable[[pd.DataFrame], pd.Series]) -> None:
        self.scorer = scorer
        self.forest = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20, random_state=42, n_jobs=-1
        )

    @staticmethod
    def _features(lineup, recent):
        features = recent.copy()
        features["number"] = pd.to_numeric(lineup.get("number"), errors="coerce")
        features["is_substitute"] = substitute_status(lineup)
        # A debut has zero appearances. A missing mean is distinct from zero
        # tackles or zero points in a player's recorded appearances.
        features["appearances"] = features["appearances"].fillna(0)
        return features.astype(float)

    def fit(self, history: RugbyDataset) -> None:
        rows = history.performances
        if rows.empty:
            raise ValueError("Cannot fit on empty history")
        dates = history.matches.set_index("match_id")["date"]
        rows = rows.assign(
            date=rows["match_id"].map(dates).dt.normalize(),
            points=score_performances(rows, self.scorer).to_numpy(),
        ).sort_values(["date", "match_id"]).reset_index(drop=True)

        stats = ["points", "minutes_played", "carries", "completed_tackles", "carried_meters"]
        available = [stat for stat in stats if stat in rows]
        players = rows.groupby("player_id", sort=False)
        # Each summary includes the appearance just played.
        means = players[available].rolling(5, min_periods=1).mean().droplevel("player_id")
        summaries = rows[["player_id", "date"]].join(means)
        summaries["appearances"] = players.cumcount() + 1

        # For training, look up the last summary strictly before each match day.
        prior = pd.merge_asof(
            rows[["player_id", "date"]], summaries, on="date", by="player_id",
            allow_exact_matches=False,
        ).drop(columns=["player_id", "date"])
        self.forest.fit(self._features(rows, prior), rows["points"])
        self._latest = summaries.drop_duplicates("player_id", keep="last").set_index("player_id")
        self._latest = self._latest.drop(columns="date")

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "_latest"):
            raise RuntimeError("Call fit() before predict()")
        if fixtures.empty:
            return pd.Series(dtype=float, index=fixtures.index, name="predicted")
        recent = self._latest.reindex(fixtures["player_id"])
        recent.index = fixtures.index
        predictions = self.forest.predict(self._features(fixtures, recent))
        return pd.Series(predictions, index=fixtures.index, name="predicted")
