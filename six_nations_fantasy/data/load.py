"""Loading the current processed Parquet tables without notebook dependencies."""

from pathlib import Path

import pandas as pd

from .dataset import RugbyDataset


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "rebuild"


# Competition IDs assigned by notebooks/scraping/rugbypass_urls.ipynb.
_RUGBYPASS_COMPETITIONS = {
    0: "Six Nations",
    1: "URC",
    2: "Top 14",
    3: "Prem",
    4: "Champions Cup",
}


def load_dataset(directory: str | Path | None = None) -> RugbyDataset:
    """Read matches.parquet and player_performances.parquet from *directory*.

    By default, use ``data/rebuild/`` at the repository root, regardless of
    the current working directory. Pass an explicit directory when loading a
    different dataset or running outside the source checkout. Relative explicit
    paths are interpreted from the current working directory.

    Both files are required; other Parquet files are left untouched.
    """
    path = DATA_DIR if directory is None else Path(directory)
    paths = [path / "matches.parquet", path / "player_performances.parquet"]
    missing = [str(file) for file in paths if not file.is_file()]
    if missing:
        raise FileNotFoundError("Required dataset file(s) missing: " + ", ".join(missing))

    matches = pd.read_parquet(paths[0])
    if "competition" not in matches and "competition_id" in matches:
        matches = matches.assign(
            competition=matches["competition_id"].map(_RUGBYPASS_COMPETITIONS)
        )

    return RugbyDataset(
        matches=matches,
        performances=pd.read_parquet(paths[1]),
    )
