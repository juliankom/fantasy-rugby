"""Historical mean with shrinkage toward an announced position's scoring average."""

from math import isfinite
from typing import Callable

import pandas as pd

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.data.lineup import substitute_status
from six_nations_fantasy.score import score_performances

from ._mean import MeanScores


# Group starting shirts into fantasy positions. Bench shirts do not identify
# which position the replacement will play, so substitutes share one target.
STARTING_POSITIONS = dict(enumerate([
    "Prop", "Hooker", "Prop", "Second-row", "Second-row",
    "Back-row", "Back-row", "Back-row", "Scrum-half", "Fly-half",
    "Back Three", "Centre", "Centre", "Back Three", "Back Three",
], start=1))


def _positions(frame, roles):
    positions = pd.Series(pd.NA, index=frame.index, dtype="string")
    if "number" in frame:
        positions = pd.to_numeric(frame["number"], errors="coerce").map(STARTING_POSITIONS)
    if roles is not None:
        positions = positions.mask(roles.fillna(False), "Substitute")
    return positions


class PositionShrinkageMean:
    """Blend the role-aware historical estimate with a training-set position mean."""

    def __init__(self, *, scorer: Callable[[pd.DataFrame], pd.Series], k: float = 5.0):
        if not isfinite(k) or k < 0:
            raise ValueError("k must be finite and non-negative")
        self.scorer = scorer
        # Five pseudo-appearances is an experiment default, not a fitted value.
        self.k = k

    def fit(self, history: RugbyDataset) -> None:
        rows = history.performances
        if rows.empty:
            raise ValueError("Cannot fit on empty history")
        scores = score_performances(rows, self.scorer)
        self._means = MeanScores(rows, scores)
        roles = substitute_status(rows)
        # Give every recorded match equal weight, including short bench appearances.
        self._targets = scores.groupby(_positions(rows, roles)).mean()
        self._counts = rows.groupby("player_id").size()
        self._role_counts = (
            scores.groupby([rows["player_id"], roles]).size() if roles is not None else None
        )

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "_means"):
            raise RuntimeError("Call fit() before predict()")
        roles = substitute_status(fixtures)
        if roles is not None:
            fixtures = fixtures.assign(is_substitute=roles)
        baseline = self._means.predict(fixtures)
        if self.k == 0:
            return baseline

        players = fixtures["player_id"]
        counts = players.map(self._counts)
        target = _positions(fixtures, roles).map(self._targets)
        if roles is not None and self._role_counts is not None:
            keys = pd.MultiIndex.from_arrays([players, roles])
            same_role = self._role_counts.reindex(keys).to_numpy()
            # The baseline transfers opposite-role history when same-role history
            # is absent. Give each transferred appearance full weight for now.
            counts = pd.Series(same_role, index=fixtures.index).fillna(counts)
            target = target.fillna(roles.map(self._means.role_means))

        # Missing position falls back to role, then the overall training mean.
        target = target.fillna(self._means.global_mean)
        counts = counts.fillna(0)
        weight = counts / (counts + self.k)
        return (weight * baseline.fillna(target) + (1 - weight) * target).rename("predicted")
