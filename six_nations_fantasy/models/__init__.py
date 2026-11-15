"""Fantasy point prediction models."""

from .historical_mean import HistoricalMean
from .recency_weighted_mean import RecencyWeightedMean

__all__ = ["HistoricalMean", "RecencyWeightedMean"]
