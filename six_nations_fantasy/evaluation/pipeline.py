"""Leakage-conscious point-prediction evaluation for historical matches."""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from six_nations_fantasy.data import RugbyDataset, TemporalSplits

from .scoring import score_performances

# The fixture rows intentionally exclude statistics observed during the match.
CONTEXT_COLUMNS = ("match_id", "player_id", "date")


class PointPredictor(Protocol):
    """A model predicting fantasy points for players in upcoming matches."""

    def fit(self, history: RugbyDataset) -> None:
        """Learn from complete historical matches and player performances.

        A model may calculate historical fantasy scores using its own scorer,
        or learn directly from raw match statistics without scoring them.
        """

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        """Predict one score per pre-match fixture row, in input row order."""


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics and one row per player-match prediction."""

    predictions: pd.DataFrame
    mae: float
    rmse: float
    bias: float
    n: int


def _context(dataset: RugbyDataset) -> pd.DataFrame:
    """Construct model input from approved pre-match identifiers only."""
    if "player_id" not in dataset.performances.columns:
        raise ValueError("performances: missing required 'player_id' column")
    if dataset.performances["player_id"].isna().any():
        raise ValueError("performances: player_id must not contain missing values")
    # Restrict columns here rather than dropping a blacklist of outcome fields.
    context = dataset.performances[["match_id", "player_id"]].merge(
        dataset.matches[["match_id", "date"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    if context.duplicated(["match_id", "player_id"]).any():
        raise ValueError("performances: duplicate player_id within a match")
    return context.loc[:, list(CONTEXT_COLUMNS)]


def evaluate_holdout(
    model: PointPredictor,
    training: RugbyDataset,
    holdout: RugbyDataset,
    *,
    scorer: Callable[[pd.Series], float],
) -> EvaluationResult:
    """Fit on completed historical matches and evaluate on later matches.

    fit() receives a RugbyDataset with complete historical match metadata and
    performance statistics, without precomputed fantasy points. A model can
    calculate scores from that history if its approach needs them.

    predict() receives only match_id, player_id and date for each evaluation
    appearance. Holdout outcomes are scored by the evaluator only after
    prediction; they are never passed into fit() or predict().

    Both training and holdout must contain matches and player performances.
    An empty holdout may result from a date range with no matches or from
    filtering to a competition absent in that period. Without performances,
    MAE and RMSE have nothing to compare against.

    Training matches must precede all holdout matches. The caller controls
    which competitions are included in each partition.
    """
    if training.matches.empty or training.performances.empty:
        raise ValueError("Training dataset must contain matches and performances")
    if holdout.matches.empty or holdout.performances.empty:
        raise ValueError("Holdout dataset must contain matches and performances")
    if training.matches["date"].max() >= holdout.matches["date"].min():
        raise ValueError("Training matches must be strictly earlier than holdout matches")
    if set(training.matches["match_id"]) & set(holdout.matches["match_id"]):
        raise ValueError("A match appears in both training and holdout")

    # Check player-match identity in both partitions before calling the model.
    _context(training)
    fixtures = _context(holdout)

    # Give fit its own copy: models may transform history during training.
    model.fit(RugbyDataset(training.matches, training.performances))
    predicted = model.predict(fixtures.copy())
    if not isinstance(predicted, pd.Series):
        raise TypeError("predict() must return a pandas Series")
    if len(predicted) != len(fixtures) or not predicted.index.equals(fixtures.index):
        raise ValueError("Predictions must align with fixture rows by index and length")
    try:
        pred = pd.to_numeric(predicted, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions must be numeric") from exc
    if not np.isfinite(pred).all():
        raise ValueError("Predictions must be finite and non-missing")

    actual = score_performances(holdout.performances, scorer).to_numpy()
    error = pred - actual
    output = fixtures.assign(actual=actual, predicted=pred, error=error)
    return EvaluationResult(
        predictions=output,
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(np.mean(error)),
        n=len(error),
    )


def evaluate_splits(
    model_factory: Callable[[], PointPredictor],
    splits: TemporalSplits,
    *,
    scorer: Callable[[pd.Series], float],
    competition: str | None = None,
) -> dict[str, EvaluationResult]:
    """Evaluate validation, then test with history extended through validation.

    Each period receives a fresh model instance. Training always uses every
    competition provided in the split; `competition` filters evaluation rows
    only. The evaluation scorer must match any scorer used by models that
    learn directly from historical fantasy points. Do not tune on test results.
    """
    validation = (splits.validation.filter_competitions([competition])
                  if competition else splits.validation)
    test = splits.test.filter_competitions([competition]) if competition else splits.test

    validation_result = evaluate_holdout(
        model_factory(), splits.train, validation, scorer=scorer
    )
    extended_training = RugbyDataset(
        pd.concat([splits.train.matches, splits.validation.matches], ignore_index=True),
        pd.concat([splits.train.performances, splits.validation.performances],
                  ignore_index=True),
    )
    test_result = evaluate_holdout(
        model_factory(), extended_training, test, scorer=scorer
    )
    return {"validation": validation_result, "test": test_result}
