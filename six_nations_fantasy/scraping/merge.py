"""Append a RugbyPass batch to a new dataset directory."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ..data import load_dataset


def merge_scrapes(base, batch, output):
    """Keep existing matches and registry names, and retain unresolved failures."""
    base, batch, output = map(Path, (base, batch, output))
    if output.exists():
        raise FileExistsError(output)

    def read(directory, table):
        return pd.read_parquet(directory / f"{table}.parquet")

    base_data = load_dataset(base)
    batch_data = load_dataset(batch) if (batch / "matches.parquet").is_file() else None
    matches = base_data.matches.set_index("match_id")
    tables = {
        "player_performances": base_data.performances,
        "team_performances": read(base, "team_performances"),
        "players": read(base, "players"),
    }
    new_ids = pd.Index([], name="match_id")
    if batch_data is not None:
        added_matches = batch_data.matches.set_index("match_id")
        new_ids = added_matches.index.difference(matches.index, sort=False)
        matches = pd.concat([matches, added_matches.loc[new_ids]], verify_integrity=True)

        players = batch_data.performances
        new_players = players.loc[players["match_id"].isin(new_ids), "player_id"]
        registry = read(batch, "players")
        tables["players"] = pd.concat([
            tables["players"], registry.loc[registry.index.isin(new_players)]
        ])
        for table, added in (
            ("player_performances", players),
            ("team_performances", read(batch, "team_performances")),
        ):
            tables[table] = pd.concat(
                [tables[table], added.loc[added["match_id"].isin(new_ids)]],
                ignore_index=True,
            )

    registry = tables["players"]
    conflicts = registry.groupby(level=0)["player_identifier"].nunique(dropna=False).gt(1)
    if conflicts.any():
        raise ValueError(f"Conflicting profiles for player IDs {conflicts[conflicts].index[:5].tolist()}")
    registry = registry.loc[~registry.index.duplicated(keep="first")]
    if not tables["player_performances"]["player_id"].isin(registry.index).all():
        raise ValueError("A performance references an unknown player")
    shared_slugs = registry["player_identifier"].duplicated(keep=False)
    if shared_slugs.any():
        raise ValueError(f"Profile slugs map to multiple player IDs: "
                         f"{registry.loc[shared_slugs, 'player_identifier'].unique()[:5].tolist()}")

    for table, key in (
        ("player_performances", ["match_id", "player_id"]),
        ("team_performances", ["match_id", "home"]),
    ):
        if tables[table].duplicated(key).any():
            raise ValueError(f"{table} contains duplicate {key} keys")
        if set(tables[table]["match_id"]) != set(matches.index):
            raise ValueError(f"{table} and matches contain different match IDs")

    tables["matches"] = matches
    tables["players"] = registry
    tables["player_performances"] = tables["player_performances"].reset_index(
        drop=True
    ).rename_axis("perf_id")
    failures = read(batch, "failures")
    if (base / "failures.parquet").is_file():
        failures = pd.concat([read(base, "failures"), failures], ignore_index=True)
    tables["failures"] = failures.loc[
        ~failures["match_id"].isin(matches.index)
    ].drop_duplicates("match_id", keep="last")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "merged"
        staged.mkdir()
        for name, table in tables.items():
            table.to_parquet(staged / f"{name}.parquet")
        staged.rename(output)
    return len(new_ids)
