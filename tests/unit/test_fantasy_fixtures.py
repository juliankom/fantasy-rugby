import pandas as pd
import pytest

from scripts.prepare_fixtures import prepare_fixtures
from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.data.fantasy_fixtures import tournament_rounds
from six_nations_fantasy.evaluation.selection import prediction_lineup, scored_player_fixtures, summarize
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
    rounds = dict(tournament_rounds(dataset.matches, 2025))
    first_round = dataset.select_matches(rounds[1]["match_id"])
    assert scored_player_fixtures(first_round, 1, roster).empty
    second_round = dataset.select_matches(rounds[2]["match_id"])
    scored = scored_player_fixtures(second_round, 2, roster)
    fixtures = prediction_lineup(scored)
    assert fixtures.player_id.tolist() == [8, 9]
    assert fixtures.match_id.tolist() == [1, 1]
    assert fixtures.is_substitute.tolist() == [False, True]
    assert "actual" not in fixtures
    assert scored.actual.tolist() == [3, 0]


def test_season_metrics_weight_players_and_rounds_correctly():
    result = summarize([
        dict(model="mean", players=1, squared_error=16, squad_points=50, optimum=100),
        dict(model="mean", players=3, squared_error=0, squad_points=150, optimum=200),
    ]).loc["mean"]
    assert result["player_RMSE"] == 2
    assert result["% of optimum"] == pytest.approx(100 * 200 / 300)


@pytest.mark.parametrize(("round_start", "training_ids"), [
    ("2025-02-01", [1, 2, 3]),
    ("2025-03-15", [1, 2, 3, 4]),
    ("2026-02-01", [1, 2, 3, 4, 5]),
])
def test_training_expands_with_completed_matches_from_all_competitions(round_start, training_ids):
    matches = pd.DataFrame({
        "match_id": range(1, 7),
        "date": pd.to_datetime(["2020-02-01", "2024-02-01", "2024-12-20",
                                "2025-02-01", "2025-03-15", "2026-02-01"]),
        "competition": ["Six Nations", "Six Nations", "URC",
                        "Six Nations", "Six Nations", "Six Nations"],
    })
    dataset = RugbyDataset(matches, matches[["match_id"]])
    training = dataset.matches_before(pd.Timestamp(round_start))
    assert training.performances.match_id.tolist() == training_ids


@pytest.mark.parametrize("labels, expected", [
    (["Round 1", "Round 3"], [1, 3]),
    ([None, None], [1, 2]),
])
def test_incomplete_validation_uses_available_weekends(labels, expected):
    matches = pd.DataFrame({
        "competition": "Six Nations", "date": pd.to_datetime(["2024-02-03", "2024-02-24"]),
        "extra_info": labels,
    })
    with pytest.warns(UserWarning, match="2024"):
        assert list(dict(tournament_rounds(matches, 2024, price_free=True))) == expected
    with pytest.warns(UserWarning, match="found 0 match weekends"):
        assert list(tournament_rounds(matches, 2023, price_free=True)) == []
