import pandas as pd
import pytest

from scripts.prepare_fixtures import prepare_fixtures
from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.data.fantasy_fixtures import fixture_path
from six_nations_fantasy.evaluation.selection import evaluation_rounds, summarize, training_before_round
from six_nations_fantasy.score import SCORE_COMPONENTS


def test_partial_sheet_preserves_announced_withdrawal(tmp_path):
    matches = pd.DataFrame({
        "match_id": range(5), "date": pd.date_range("2025-02-01", periods=5, freq="7D"),
        "competition": "Six Nations", "home_team": "Ireland", "away_team": "France",
    })
    performances = pd.DataFrame({
        "match_id": range(5), "player_id": 8, "home": True, "number": 1,
        **{stat: 0 for stat in SCORE_COMPONENTS},
    }).assign(completed_tackles=3)
    dataset = RugbyDataset(matches, performances)
    players = pd.DataFrame({"player_id": [8, 9], "name": ["Thomas Clarkson", "Other Player"]})
    csv = tmp_path / "prices.csv"
    pd.DataFrame({
        "year": 2025, "round": 2, "team": "Ireland", "player": ["Tom Clarkson", "Other Player"],
        "nominal_position": "prop", "price": 10, "selection": ["starter", "bench"],
        "shirt_number": [1, 17],
    }).to_csv(csv, index=False)
    roster = prepare_fixtures(csv, 2025, dataset, players)
    roster.to_parquet(fixture_path(tmp_path, 2025))
    (number, round_start, fixtures, actual, pool), = evaluation_rounds(dataset, 2025, tmp_path, require_prices=True)
    assert training_before_round(dataset, round_start).matches.match_id.tolist() == [0]
    assert number == 2
    assert fixtures.player_id.tolist() == [8, 9]
    assert fixtures.match_id.tolist() == [1, 1]
    assert fixtures.is_substitute.tolist() == [False, True]
    assert "actual" not in fixtures
    assert actual.tolist() == pool.actual.tolist() == [3, 0]


def test_season_metrics_weight_players_and_rounds_correctly():
    result = summarize([
        dict(model="mean", players=1, squared_error=16, squad_points=50, optimum=100),
        dict(model="mean", players=3, squared_error=0, squad_points=150, optimum=200),
    ]).loc["mean"]
    assert result["player_RMSE"] == 2
    assert result["% of optimum"] == pytest.approx(100 * 200 / 300)


@pytest.mark.parametrize(("round_start", "training_ids"), [
    ("2025-02-01", [1, 2]),
    ("2025-03-15", [1, 2, 4]),
    ("2026-02-01", [1, 2, 4, 5]),
])
def test_training_expands_with_completed_six_nations_matches(round_start, training_ids):
    matches = pd.DataFrame({
        "match_id": range(1, 7),
        "date": pd.to_datetime(["2020-02-01", "2024-02-01", "2024-12-20",
                                "2025-02-01", "2025-03-15", "2026-02-01"]),
        "competition": ["Six Nations", "Six Nations", "URC",
                        "Six Nations", "Six Nations", "Six Nations"],
    })
    dataset = RugbyDataset(matches, matches[["match_id"]])
    training = training_before_round(dataset, pd.Timestamp(round_start))
    assert training.performances.match_id.tolist() == training_ids
