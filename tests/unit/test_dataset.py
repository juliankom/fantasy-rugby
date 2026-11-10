from datetime import date

import pandas as pd
import pytest

from six_nations_fantasy.data import RugbyDataset


@pytest.fixture
def dataset():
    matches = pd.DataFrame(
        {
            "date": [date(2024, 11, 3), date(2025, 2, 1), date(2026, 2, 1)],
            "competition": ["URC", "Six Nations", "Six Nations"],
        },
        index=pd.Index(["m1", "m2", "m3"], name="match_id"),
    )
    performances = pd.DataFrame(
        {"match_id": ["m1", "m1", "m2", "m3", "m3"], "player_id": [1, 2, 1, 2, 3]}
    )
    return RugbyDataset(matches, performances)


def test_named_match_index_becomes_column(dataset):
    assert dataset.matches["match_id"].tolist() == ["m1", "m2", "m3"]
    assert pd.api.types.is_datetime64_any_dtype(dataset.matches["date"])


def test_select_matches_keeps_all_player_rows(dataset):
    subset = dataset.select_matches(["m3", "m1"])
    assert set(subset.matches["match_id"]) == {"m1", "m3"}
    assert subset.performances["match_id"].tolist() == ["m1", "m1", "m3", "m3"]


def test_filter_competitions(dataset):
    filtered = dataset.filter_competitions(["Six Nations"])
    assert set(filtered.matches["match_id"]) == {"m2", "m3"}
    assert set(filtered.performances["match_id"]) == {"m2", "m3"}


def test_join_keeps_all_performances(dataset):
    joined = dataset.player_matches()
    assert len(joined) == len(dataset.performances)
    assert joined.loc[joined["match_id"] == "m1", "competition"].eq("URC").all()


def test_original_dataframes_not_mutated():
    matches = pd.DataFrame({"match_id": [1], "date": [date(2024, 1, 1)]})
    performances = pd.DataFrame({"match_id": [1], "player_id": [1]})
    RugbyDataset(matches, performances)
    assert matches["date"].iloc[0] == date(2024, 1, 1)
    assert "date" not in performances


@pytest.mark.parametrize(
    "matches,performances,error",
    [
        (
            pd.DataFrame({"match_id": ["m", "m"], "date": ["2025-01-01"] * 2}),
            pd.DataFrame({"match_id": ["m"]}),
            "unique",
        ),
        (
            pd.DataFrame({"match_id": ["m"], "date": ["2025-01-01"]}),
            pd.DataFrame({"match_id": ["other"]}),
            "not found",
        ),
        (
            pd.DataFrame({"match_id": ["m"], "date": [None]}),
            pd.DataFrame({"match_id": ["m"]}),
            "date",
        ),
        (
            pd.DataFrame({"match_id": ["m"], "date": ["not a date"]}),
            pd.DataFrame({"match_id": ["m"]}),
            "invalid dates",
        ),
        (
            pd.DataFrame({"date": ["2025-01-01"]}),
            pd.DataFrame({"match_id": ["m"]}),
            "match_id",
        ),
        (
            pd.DataFrame({"match_id": ["m"], "date": ["2025-01-01"]}),
            pd.DataFrame({"player_id": [123]}),
            "match_id",
        ),
    ],
)
def test_invalid_tables_raise_clear_errors(matches, performances, error):
    with pytest.raises(ValueError, match=error):
        RugbyDataset(matches, performances)
