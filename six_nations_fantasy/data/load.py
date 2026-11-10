"""Loading the current processed Parquet tables without notebook dependencies."""

from pathlib import Path

import pandas as pd

from .dataset import RugbyDataset


_DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_dataset(directory: str | Path | None = None) -> RugbyDataset:
    """Read matches.parquet and player_performances.parquet from *directory*.

    By default, use ``data/processed/`` at the repository root, regardless of
    the current working directory. Pass an explicit directory when loading a
    different dataset or running outside the source checkout. Relative explicit
    paths are interpreted from the current working directory.

    Both files are required; other Parquet files are left untouched.
    """
    path = _DEFAULT_DATA_DIRECTORY if directory is None else Path(directory)
    paths = [path / "matches.parquet", path / "player_performances.parquet"]
    missing = [str(file) for file in paths if not file.is_file()]
    if missing:
        raise FileNotFoundError("Required dataset file(s) missing: " + ", ".join(missing))

    return RugbyDataset(
        matches=pd.read_parquet(paths[0]),
        performances=pd.read_parquet(paths[1]),
    )
