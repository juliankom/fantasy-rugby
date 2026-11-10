"""Chronological train/validation/test partitions with match-level isolation."""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from .dataset import RugbyDataset


@dataclass(frozen=True)
class TemporalSplits:
    """Non-overlapping match-level partitions of one historical dataset."""

    train: RugbyDataset
    validation: RugbyDataset
    test: RugbyDataset


def _cutoff(value: str | date | datetime | pd.Timestamp, label: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a valid date") from exc
    if pd.isna(ts):
        raise ValueError(f"{label}: date must not be missing")
    if ts.tzinfo is not None:
        raise ValueError(f"{label}: date must be timezone-naive")
    if ts != ts.normalize():
        raise ValueError(f"{label}: expected a date without a time component")
    return ts


def temporal_split(
    dataset: RugbyDataset,
    *,
    validation_start: str | date | datetime | pd.Timestamp,
    test_start: str | date | datetime | pd.Timestamp,
) -> TemporalSplits:
    """Split by match date using half-open, chronological windows.

    train:       match date < validation_start
    validation:  validation_start <= match date < test_start
    test:        match date >= test_start

    All player performances inherit their match's partition. The method does
    not shuffle, sample, fit statistics, or use future observations.
    """
    validation_date = _cutoff(validation_start, "validation_start")
    test_date = _cutoff(test_start, "test_start")
    if validation_date >= test_date:
        raise ValueError("validation_start must be earlier than test_start")

    dates = dataset.matches["date"]
    ids = dataset.matches["match_id"]
    return TemporalSplits(
        train=dataset.select_matches(ids.loc[dates < validation_date]),
        validation=dataset.select_matches(
            ids.loc[(dates >= validation_date) & (dates < test_date)]
        ),
        test=dataset.select_matches(ids.loc[dates >= test_date]),
    )
