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
