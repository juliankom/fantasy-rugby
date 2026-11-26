from pathlib import Path

from datetime import date

import pandas as pd
import pytest

from six_nations_fantasy.data import load_dataset


def test_missing_files_listed(tmp_path):
    with pytest.raises(FileNotFoundError, match="matches.parquet") as exc:
        load_dataset(tmp_path)
    assert "player_performances.parquet" in str(exc.value)


def test_loader_reads_both_tables(tmp_path, monkeypatch):
    (tmp_path / "matches.parquet").touch()
    (tmp_path / "player_performances.parquet").touch()
    matches = pd.DataFrame({"match_id": [1], "date": [date(2025, 1, 1)]})
    performances = pd.DataFrame({"match_id": [1], "player_id": [9]})
    looked_up = []

    def fake_read_parquet(path):
        looked_up.append(path.name)
        return {"matches.parquet": matches, "player_performances.parquet": performances}[path.name]

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    dataset = load_dataset(tmp_path)
    assert looked_up == ["matches.parquet", "player_performances.parquet"]
    assert dataset.performances["player_id"].tolist() == [9]


def test_default_directory_is_source_checkout_data_directory():
    from six_nations_fantasy.data import load as load_module

    assert load_module.DATA_DIR == (
        Path(load_module.__file__).resolve().parents[2] / "data" / "default"
    )


def test_default_directory_is_independent_of_working_directory(tmp_path, monkeypatch):
    from six_nations_fantasy.data import load as load_module

    processed = tmp_path / "data" / "default"
    processed.mkdir(parents=True)
    (processed / "matches.parquet").touch()
    (processed / "player_performances.parquet").touch()
    other_directory = tmp_path / "elsewhere"
    other_directory.mkdir()

    monkeypatch.setattr(load_module, "DATA_DIR", processed)
    monkeypatch.chdir(other_directory)

    matches = pd.DataFrame({"match_id": [1], "date": [date(2025, 1, 1)]})
    performances = pd.DataFrame({"match_id": [1], "player_id": [9]})
    observed_paths = []

    def fake_read_parquet(path):
        observed_paths.append(path)
        return {
            "matches.parquet": matches,
            "player_performances.parquet": performances,
        }[path.name]

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    result = load_dataset()

    assert observed_paths == [
        processed / "matches.parquet",
        processed / "player_performances.parquet",
    ]
    assert result.performances["player_id"].tolist() == [9]


def test_loader_resolves_processed_competition_ids(tmp_path):
    matches = pd.DataFrame({
        "match_id": [1, 2],
        "date": [date(2026, 2, 1), date(2026, 2, 1)],
        "competition_id": [0, 1],
    })
    performances = pd.DataFrame({"match_id": [1, 2], "player_id": [9, 10]})
    matches.to_parquet(tmp_path / "matches.parquet")
    performances.to_parquet(tmp_path / "player_performances.parquet")

    tournament = load_dataset(tmp_path).filter_competitions(["Six Nations"])

    assert tournament.matches["match_id"].tolist() == [1]
    assert tournament.performances["player_id"].tolist() == [9]
