"""Predict total fantasy points with component-specific shrinkage."""

from collections.abc import Mapping

import pandas as pd

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.score import SCORE_COMPONENTS

from .position_shrinkage import PositionShrinkageMean


class ComponentShrinkageMean:
    """Apply position shrinkage to each scoring component and sum its forecast.

    k is the shared default. k_by_stat overrides it by raw stat name.
    A common k recovers position shrinkage applied to the total score.
    """

    def __init__(self, *, k: float = 20, k_by_stat: Mapping[str, float] | None = None):
        overrides = dict(k_by_stat or {})
        unknown = overrides.keys() - SCORE_COMPONENTS.keys()
        if unknown:
            raise ValueError(f"Unknown scoring components: {sorted(unknown)}")
        self.models = {
            stat: PositionShrinkageMean(k=overrides.get(stat, k), scorer=scorer)
            for stat, scorer in SCORE_COMPONENTS.items()
        }

    def fit(self, history: RugbyDataset) -> None:
        for model in self.models.values():
            model.fit(history)

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        return sum(model.predict(fixtures) for model in self.models.values()).rename("predicted")
