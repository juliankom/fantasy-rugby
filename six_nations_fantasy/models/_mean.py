"""Shared lineup conditioning for the mean-score models."""

import pandas as pd

from six_nations_fantasy.data.lineup import substitute_status


class MeanScores:
    """Fit player and lineup-role means using the supplied match weights."""

    def __init__(
        self, performances: pd.DataFrame, scores: pd.Series, weights: pd.Series | None = None
    ) -> None:
        history = pd.DataFrame({
            "player_id": performances["player_id"].to_numpy(),
            "weighted_score": scores.to_numpy(),
            "weight": 1.0 if weights is None else weights.to_numpy(),
        })
        history["weighted_score"] *= history["weight"]

        # compute the (weighted) means per-player as well as globally
        self.player_means = self._average(history, "player_id")
        self.global_mean = history["weighted_score"].sum() / history["weight"].sum()

        # additionally, compute the same values, but this time separated by whether 
        # the player was a substitute or not 
        roles = substitute_status(performances)
        if roles is not None:
            history["is_substitute"] = roles.array
        self.role_means = self._average(history, "is_substitute") if roles is not None else None
        self.player_role_means = (
            self._average(history, ["player_id", "is_substitute"]) if roles is not None else None
        )

    @staticmethod
    def _average(history: pd.DataFrame, by: str | list[str]) -> pd.Series:
        sums = history.groupby(by)[["weighted_score", "weight"]].sum()
        return sums["weighted_score"] / sums["weight"]

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        player_ids = fixtures["player_id"]
        predicted = player_ids.map(self.player_means).fillna(self.global_mean)

        if self.role_means is not None and "is_substitute" in fixtures:
            roles = fixtures["is_substitute"].astype("boolean")
            keys = pd.MultiIndex.from_arrays([player_ids, roles])
            same_role = pd.Series(self.player_role_means.reindex(keys).to_numpy(), index=fixtures.index)
            opposite_roles = ~roles
            opposite_keys = pd.MultiIndex.from_arrays([player_ids, opposite_roles])
            opposite_role = pd.Series(
                self.player_role_means.reindex(opposite_keys).to_numpy(), index=fixtures.index
            )
            pooled = roles.map(self.role_means)

            # Keep the player's difference from the pool when extrapolating to a new role.
            # This is just an arbitrary choice an can be fiddled with if desired
            adjusted = opposite_role + (pooled - opposite_roles.map(self.role_means))

            # Use the player's history in this role unless they don't have one, in which case 
            # use the adjusted prediction or fall back to the global mean for the role
            predicted = same_role.fillna(adjusted).fillna(pooled)
        return predicted.rename("predicted")
