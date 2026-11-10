"""Model-independent evaluation on chronological rugby match datasets."""

from .baselines import HistoricalMean, RecencyWeightedMean
from .pipeline import EvaluationResult, PointPredictor, evaluate_holdout, evaluate_splits

__all__ = [
    "EvaluationResult",
    "HistoricalMean",
    "PointPredictor",
    "RecencyWeightedMean",
    "evaluate_holdout",
    "evaluate_splits",
]
