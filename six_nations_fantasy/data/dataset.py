"""Validated, match-oriented views of rugby performance data."""

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


def _require_key(frame: pd.DataFrame, key: str, table: str) -> pd.DataFrame:
    """Accept an explicit key column or a named index, never infer row numbers."""
    if key in frame.columns:
        if frame.index.name == key:
            raise ValueError(f"{table}: {key!r} is both an index and a column")
        return frame.copy()
    if frame.index.name == key:
        return frame.reset_index()
    raise ValueError(f"{table}: expected {key!r} as a column or named index")


@dataclass(frozen=True)
class RugbyDataset:
    """One match table and its player performances, with foreign keys checked.

    Each instance owns copies of its DataFrames. The original performance columns
    are preserved; only the match date is normalized to pandas datetime.
    """

    matches: pd.DataFrame
    performances: pd.DataFrame

    def __post_init__(self) -> None:
        matches = _require_key(self.matches, "match_id", "matches")
        performances = _require_key(self.performances, "match_id", "performances")

        if "date" not in matches.columns:
            raise ValueError("matches: missing required 'date' column")
        if matches["match_id"].isna().any() or not matches["match_id"].is_unique:
            raise ValueError("matches: match_id must be non-null and unique")
        if performances["match_id"].isna().any():
            raise ValueError("performances: match_id must be non-null")

        try:
            dates = pd.to_datetime(matches["date"], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("matches: 'date' contains invalid dates") from exc
        if dates.isna().any():
            raise ValueError("matches: 'date' must not contain missing values")
        if isinstance(dates.dtype, pd.DatetimeTZDtype):
            raise ValueError("matches: 'date' must be timezone-naive")
        matches["date"] = dates

        unknown = ~performances["match_id"].isin(matches["match_id"])
        if unknown.any():
            bad = performances.loc[unknown, "match_id"].drop_duplicates().head(5).tolist()
            raise ValueError(f"performances: match IDs not found in matches: {bad}")

        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "performances", performances)

    def select_matches(self, match_ids: Iterable) -> "RugbyDataset":
        """Return matches and their performances together; never split a match."""
        ids = set(match_ids)
        return RugbyDataset(
            matches=self.matches.loc[self.matches["match_id"].isin(ids)],
            performances=self.performances.loc[
                self.performances["match_id"].isin(ids)
            ],
        )

    def filter_competitions(self, competitions: Iterable[str]) -> "RugbyDataset":
        """Filter a partition to specific competitions (e.g. Six Nations)."""
        if "competition" not in self.matches.columns:
            raise ValueError("matches: no 'competition' column available to filter")
        competition_names = set(competitions)
        selected = self.matches.loc[
            self.matches["competition"].isin(competition_names), "match_id"
        ]
        return self.select_matches(selected)

    def player_matches(self) -> pd.DataFrame:
        """Join match metadata onto each performance without losing player rows.

        Keeps existing performance fields. Conflicting match fields receive a
        '_match' suffix, so the original player column is not overwritten.
        """
        return self.performances.merge(
            self.matches,
            on="match_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_match"),
        )
