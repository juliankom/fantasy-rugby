"""Leakage-conscious point-prediction evaluation for historical matches."""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from six_nations_fantasy.data import RugbyDataset, TemporalSplits

# Model-facing rows deliberately do not contain any performance statistics.
CONTEXT_COLUMNS = ("match_id", "player_id", "date")
TARGET_COLUMN = "fantasy_points"


class PointPredictor(Protocol):
    """A predictor of expected fantasy points for a player in a given match."""

    def fit(self, history: pd.DataFrame) -> None:
        """Fit on historical context and `fantasy_points` targets."""

    def predict(self, fixtures: pd.DataFrame) -> pd.Series:
        """Predict one score per fixture row, in the original row order."""


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


def _targets(dataset: RugbyDataset, scorer: Callable[[pd.Series], float]) -> pd.Series:
    """Compute labels from outcomes; never pass these rows to predict()."""
    try:
        values = dataset.performances.apply(scorer, axis=1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not calculate fantasy points from performances: {exc}") from exc
    try:
        result = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scorer must return numerical fantasy points") from exc
    if not np.isfinite(result).all():
        raise ValueError("Scorer returned missing or non-finite fantasy points")
    return pd.Series(result, name=TARGET_COLUMN)


def evaluate_holdout(
    model: PointPredictor,
    training: RugbyDataset,
    holdout: RugbyDataset,
    *,
    scorer: Callable[[pd.Series], float],
) -> EvaluationResult:
    """Fit on past matches and evaluate on a strictly later holdout.

    The model sees only player_id, match_id and match date at prediction time.
    The scorer is run on the holdout only **after** predict() returns.
    The caller controls which competitions are included in each partition.
    """
    if training.matches.empty or training.performances.empty:
        raise ValueError("Training dataset must contain matches and performances")
    if holdout.matches.empty or holdout.performances.empty:
        raise ValueError("Holdout dataset must contain matches and performances")
    if training.matches["date"].max() >= holdout.matches["date"].min():
        raise ValueError("Training matches must be strictly earlier than holdout matches")
    if set(training.matches["match_id"]) & set(holdout.matches["match_id"]):
        raise ValueError("A match appears in both training and holdout")

    history = _context(training)
    history[TARGET_COLUMN] = _targets(training, scorer).to_numpy()
    fixtures = _context(holdout)

    model.fit(history.copy())
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

    actual = _targets(holdout, scorer).to_numpy()
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
    only. No parameter tuning should be performed using the test results.
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
