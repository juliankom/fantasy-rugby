"""Loading the current processed Parquet tables without notebook dependencies."""

from pathlib import Path

import pandas as pd

from .dataset import RugbyDataset


def load_dataset(directory: str | Path = "notebooks/data") -> RugbyDataset:
    """Read matches.parquet and player_performances.parquet from *directory*.

    The default is the repository's existing data location, relative to the
    current working directory. Pass an explicit directory from elsewhere.
    Both files are required; other Parquet files are left untouched.
    """
    path = Path(directory)
    paths = [path / "matches.parquet", path / "player_performances.parquet"]
    missing = [str(file) for file in paths if not file.is_file()]
    if missing:
        raise FileNotFoundError("Required dataset file(s) missing: " + ", ".join(missing))

    return RugbyDataset(
        matches=pd.read_parquet(paths[0]),
        performances=pd.read_parquet(paths[1]),
    )
