from datetime import date

import pandas as pd
import pytest

from six_nations_fantasy.data import RugbyDataset, temporal_split
from six_nations_fantasy.evaluation import (
    HistoricalMean,
    RecencyWeightedMean,
    evaluate_holdout,
)


def fake_score(row):
    return row["points"]


@pytest.fixture
def dataset():
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4", "m5"],
            "date": [date(2024, 2, 1), date(2024, 8, 1), date(2025, 2, 1),
                     date(2026, 2, 1), date(2026, 3, 1)],
            "competition": ["URC", "URC", "Six Nations", "Six Nations", "URC"],
        }
    )
    performances = pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m2", "m3", "m3", "m4", "m4", "m5"],
            "player_id": [1, 2, 1, 1, 3, 1, 4, 2],
            "points": [10, 30, 20, 40, 50, 25, 5, 60],
            "tries": [0, 2, 0, 1, 2, 0, 0, 1],
        }
    )
    return RugbyDataset(matches, performances)


def test_evaluate_predictions_and_metrics(dataset):
    split = temporal_split(dataset, validation_start="2025-01-01", test_start="2026-01-01")
    result = evaluate_holdout(HistoricalMean(), split.train, split.validation, scorer=fake_score)
    assert result.n == 2
    assert result.predictions["player_id"].tolist() == [1, 3]
    # Player 1 trained on 10,20 -> 15. Player 3 unseen -> global mean of 10,30,20 = 20
    assert result.predictions["predicted"].tolist() == pytest.approx([15, 20])
    assert result.predictions["actual"].tolist() == [40, 50]
    assert result.mae == pytest.approx(27.5)
    assert result.rmse == pytest.approx(((25**2 + 30**2) / 2) ** 0.5)
    assert result.bias == pytest.approx(-27.5)


def test_predict_does_not_receive_outcome_columns(dataset):
    split = temporal_split(dataset, validation_start="2025-01-01", test_start="2026-01-01")
    seen = {}

    class Spy:
        def fit(self, history):
            seen["fit"] = set(history.columns)

        def predict(self, fixtures):
            seen["predict"] = set(fixtures.columns)
            return pd.Series([42.0] * len(fixtures), index=fixtures.index)

    result = evaluate_holdout(Spy(), split.train, split.validation, scorer=fake_score)
    assert seen["fit"] == {"player_id", "match_id", "date", "fantasy_points"}
    assert seen["predict"] == {"player_id", "match_id", "date"}
    assert "tries" not in seen["predict"]
    assert result.n == 2


def test_rejects_overlapping_dates_even_for_different_matches(dataset):
    a = dataset.select_matches(["m1", "m3"])
    b = dataset.select_matches(["m2"])
    with pytest.raises(ValueError, match="strictly earlier"):
        evaluate_holdout(HistoricalMean(), a, b, scorer=fake_score)


def test_rejects_empty_holdout(dataset):
    with pytest.raises(ValueError, match="Holdout"):
        evaluate_holdout(HistoricalMean(), dataset, dataset.select_matches([]), scorer=fake_score)


def test_rejects_duplicate_player_match(dataset):
    perf = pd.concat([dataset.performances, dataset.performances.iloc[:1]], ignore_index=True)
    broken = RugbyDataset(dataset.matches, perf)
    with pytest.raises(ValueError, match="duplicate player_id"):
        evaluate_holdout(HistoricalMean(), broken.select_matches(["m1"]),
                         broken.select_matches(["m3"]), scorer=fake_score)


@pytest.mark.parametrize("predict", [
    lambda frame: pd.Series([1.0]),
    lambda frame: pd.Series([float("nan")] * len(frame), index=frame.index),
    lambda frame: [1.0] * len(frame),
])
def test_rejects_bad_predictions(dataset, predict):
    split = temporal_split(dataset, validation_start="2025-01-01", test_start="2026-01-01")

    class Bad:
        def fit(self, history):
            pass

        def predict(self, fixtures):
            return predict(fixtures)

    with pytest.raises((TypeError, ValueError)):
        evaluate_holdout(Bad(), split.train, split.validation, scorer=fake_score)


def test_competition_filter_applies_only_to_holdout(dataset):
    split = temporal_split(dataset, validation_start="2025-01-01", test_start="2026-01-01")
    history = dataset.select_matches(["m1", "m2", "m3"])
    holdout = split.test.filter_competitions(["Six Nations"])
    result = evaluate_holdout(HistoricalMean(), history, holdout, scorer=fake_score)
    assert result.n == 2
    assert result.predictions["match_id"].tolist() == ["m4", "m4"]


def test_recency_weighted_mean_prefers_newer_match():
    h = pd.DataFrame({
        "player_id": [1, 1],
        "date": pd.to_datetime(["2025-01-01", "2025-01-11"]),
        "fantasy_points": [10.0, 30.0],
    })
    model = RecencyWeightedMean(half_life_days=10)
    model.fit(h)
    result = model.predict(pd.DataFrame({"player_id": [1, 2]}))
    assert result.tolist() == pytest.approx([(10 * .5 + 30) / 1.5, (10 * .5 + 30) / 1.5])


def test_historical_mean_requires_fit():
    with pytest.raises(RuntimeError, match="fit"):
        HistoricalMean().predict(pd.DataFrame({"player_id": [1]}))


def test_evaluate_splits_retrains_on_validation_but_not_test(dataset):
    from six_nations_fantasy.evaluation import evaluate_splits

    split = temporal_split(dataset, validation_start="2025-01-01", test_start="2026-01-01")
    seen_train_sizes = []

    class TrackingModel(HistoricalMean):
        def fit(self, history):
            seen_train_sizes.append(len(history))
            super().fit(history)

    results = evaluate_splits(TrackingModel, split, scorer=fake_score,
                              competition="Six Nations")
    # 3 performances before 2025; 2 more from 2025 for the 2026 retrain.
    assert seen_train_sizes == [3, 5]
    assert results["validation"].n == 2
    assert results["test"].n == 2  # m5 (URC) excluded


def test_scorer_only_sees_holdout_after_prediction(dataset):
    split = temporal_split(dataset, validation_start="2025-01-01", test_start="2026-01-01")
    events = []

    class Spy:
        def fit(self, history):
            events.append("fit")

        def predict(self, fixtures):
            events.append("predict")
            return pd.Series([10.0] * len(fixtures), index=fixtures.index)

    def score_with_order(row):
        events.append("score")
        return row["points"]

    evaluate_holdout(Spy(), split.train, split.validation, scorer=score_with_order)
    assert events[:3] == ["score", "score", "score"]
    assert events[3:5] == ["fit", "predict"]
    assert events[5:] == ["score", "score"]
