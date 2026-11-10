from datetime import date

import pandas as pd
import pytest

from six_nations_fantasy.data import RugbyDataset, temporal_split


@pytest.fixture
def dataset():
    # Two players per match; a split must never separate them.
    matches = pd.DataFrame(
        {
            "match_id": ["pre", "boundary", "val", "test", "late"],
            "date": [
                date(2024, 12, 31),
                date(2025, 1, 1),
                date(2025, 11, 2),
                date(2026, 1, 1),
                date(2026, 2, 1),
            ],
            "competition": ["URC", "URC", "Six Nations", "Six Nations", "URC"],
        }
    )
    performances = pd.DataFrame(
        {"match_id": [mid for mid in matches["match_id"] for _ in range(2)],
         "player_id": [1, 2] * len(matches)}
    )
    return RugbyDataset(matches, performances)


def test_split_is_chronological_and_match_safe(dataset):
    splits = temporal_split(
        dataset, validation_start="2025-01-01", test_start="2026-01-01"
    )
    expected = [
        (splits.train, {"pre"}),
        (splits.validation, {"boundary", "val"}),
        (splits.test, {"test", "late"}),
    ]
    for partition, match_ids in expected:
        assert set(partition.matches["match_id"]) == match_ids
        assert set(partition.performances["match_id"]) == match_ids
        assert len(partition.performances) == 2 * len(match_ids)

    # No matches or performances were lost.
    assert sum(len(s.matches) for s, _ in expected) == len(dataset.matches)
    assert sum(len(s.performances) for s, _ in expected) == len(dataset.performances)


def test_competition_filtering_after_split(dataset):
    splits = temporal_split(
        dataset, validation_start=date(2025, 1, 1), test_start=date(2026, 1, 1)
    )
    six_nations_test = splits.test.filter_competitions(["Six Nations"])
    assert six_nations_test.matches["match_id"].tolist() == ["test"]
    assert six_nations_test.performances["match_id"].tolist() == ["test", "test"]


@pytest.mark.parametrize(
    "validation_start,test_start",
    [
        ("2026-01-01", "2025-01-01"),
        ("2025-01-01", "2025-01-01"),
        ("bad-date", "2026-01-01"),
        (None, "2026-01-01"),
        ("2025-01-01", "2026-01-01T15:00"),
        ("2025-01-01T00:00:00Z", "2026-01-01"),
    ],
)
def test_rejects_invalid_boundaries(dataset, validation_start, test_start):
    with pytest.raises(ValueError):
        temporal_split(
            dataset, validation_start=validation_start, test_start=test_start
        )


def test_empty_partition_is_explicit_and_safe(dataset):
    splits = temporal_split(
        dataset, validation_start="2023-01-01", test_start="2024-01-01"
    )
    assert splits.train.matches.empty
    assert splits.train.performances.empty
