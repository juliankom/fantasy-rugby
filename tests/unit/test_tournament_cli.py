"""Checks date-window selection independently of any real Parquet files."""

from datetime import date

import pandas as pd
import pytest

from scripts.evaluate_baseline import select_evaluation_period
from six_nations_fantasy.data import RugbyDataset


@pytest.fixture
def dataset():
    return RugbyDataset(
        matches=pd.DataFrame({
            "match_id": ["old", "before", "start", "inside", "end"],
            "date": pd.to_datetime([
                "2024-01-01", "2026-01-31", "2026-02-01",
                "2026-03-01", "2026-04-01",
            ]),
            "competition": ["URC", "Six Nations", "Six Nations", "URC", "Six Nations"],
        }),
        performances=pd.DataFrame({
            "match_id": ["old", "before", "start", "start", "inside", "end"],
            "player_id": [1, 1, 1, 2, 3, 1],
        }),
    )


def test_window_is_start_inclusive_end_exclusive(dataset):
    training, evaluation = select_evaluation_period(
        dataset, start=date(2026, 2, 1), end=date(2026, 4, 1),
    )
    assert training.matches["match_id"].tolist() == ["old", "before"]
    assert evaluation.matches["match_id"].tolist() == ["start", "inside"]
    assert evaluation.performances["match_id"].tolist() == ["start", "start", "inside"]


def test_competition_filter_applies_only_to_evaluation(dataset):
    training, evaluation = select_evaluation_period(
        dataset, start=date(2026, 2, 1), end=date(2026, 4, 1),
        competition="Six Nations",
    )
    assert training.matches["match_id"].tolist() == ["old", "before"]
    assert evaluation.matches["match_id"].tolist() == ["start"]
    assert evaluation.performances["player_id"].tolist() == [1, 2]


def test_no_end_includes_all_later_matches(dataset):
    _, evaluation = select_evaluation_period(dataset, start=date(2026, 2, 1))
    assert evaluation.matches["match_id"].tolist() == ["start", "inside", "end"]


@pytest.mark.parametrize("end", [date(2026, 2, 1), date(2026, 1, 1)])
def test_rejects_invalid_end_date(dataset, end):
    with pytest.raises(ValueError, match="later than"):
        select_evaluation_period(dataset, start=date(2026, 2, 1), end=end)
