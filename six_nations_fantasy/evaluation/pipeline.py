"""Leakage-conscious point-prediction evaluation for historical matches."""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.data.lineup import substitute_status
from six_nations_fantasy.score import score_performances

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
        """Predict one score per fixture row, optionally using is_substitute."""
        return pd.Series()


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics and one row per player-match prediction."""

    predictions: pd.DataFrame
    mae: float
    rmse: float
    bias: float
    n: int


def recorded_lineup(dataset: RugbyDataset) -> pd.DataFrame:
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
    context = context.loc[:, list(CONTEXT_COLUMNS)]
    if "number" in dataset.performances:
        context["number"] = dataset.performances["number"].to_numpy()
    roles = substitute_status(dataset.performances)
    if roles is not None:
        context["is_substitute"] = roles.array
    return context


def evaluate_holdout(
    model: PointPredictor,
    training: RugbyDataset,
    holdout: RugbyDataset,
    *,
    scorer: Callable[[pd.DataFrame], pd.Series],
) -> EvaluationResult:
    """Fit on completed historical matches and evaluate on later matches.

    fit() receives a RugbyDataset with complete historical match metadata and
    performance statistics, without precomputed fantasy points. A model can
    calculate scores from that history if its approach needs them.

    predict() receives match_id, player_id, date and, when available,
    number and is_substitute from the recorded lineup for each evaluation appearance.
    Holdout outcomes are scored by the evaluator only after
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
    recorded_lineup(training)
    fixtures = recorded_lineup(holdout)

    pred = fit_predict(model, training, fixtures)
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


def fit_predict(model, training, fixtures):
    """Fit on prior matches and return aligned, finite player predictions."""
    # Give fit its own copy: models may transform history during training.
    model.fit(RugbyDataset(training.matches, training.performances))
    return predict_points(model, fixtures)


def predict_points(model, fixtures):
    """Predict with a fitted model and validate player-row alignment."""
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

    return pd.Series(pred, index=fixtures.index)
