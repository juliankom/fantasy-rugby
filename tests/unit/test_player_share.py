import numpy as np
import pandas as pd
import pytest

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.evaluation.pipeline import recorded_lineup
from six_nations_fantasy.evaluation.selection import scored_player_fixtures, prediction_lineup
from six_nations_fantasy.models import PlayerShareModel
from six_nations_fantasy.models._share_regression import TeamRate
from six_nations_fantasy.models.player_share import component_units
from six_nations_fantasy.score import POINTS_PER_UNIT


def history():
    matches, players = [], []
    for day in range(7):
        matches.append(dict(match_id=day, date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                            home_team="A", away_team="B", competition="Six Nations"))
        for home in [True, False]:
            for shirt in range(1, 24):
                row = dict.fromkeys(POINTS_PER_UNIT, 0)
                row.update(match_id=day, player_id=shirt + 23 * home, home=home, number=shirt,
                           tries=int(shirt == 5), completed_tackles=shirt % 4,
                           carried_meters=19 if shirt < 22 else -1, yellow_cards=int(shirt == 1))
                players.append(row)
    return RugbyDataset(pd.DataFrame(matches), pd.DataFrame(players))


def test_tau_shrinks_opponent_multiplier():
    teams = pd.DataFrame({"team": "A", "competition": "Cup", "home": True,
                          "opponent": ["B", "C"] * 10})
    counts = np.tile([8, 2], 10)
    ratios = []
    for tau in [0, 0.1, 1]:
        rates = TeamRate(teams, counts, tau).predict(teams)
        ratios.append(rates[0] / rates[1])
    assert ratios[0] == 1
    assert 1 < ratios[1] < ratios[2] < 4


def test_announced_lineup_allocation_and_scoring():
    dataset = history()
    held = dataset.select_matches([6])
    model = PlayerShareModel(tau=0, tau_by_stat={"tries": 0.5})
    model.fit(dataset.matches_before(held.matches.date.min()))
    recorded = recorded_lineup(held)
    roster = recorded[["match_id", "player_id", "team", "date", "number", "is_substitute"]].rename(
        columns={"number": "shirt_number", "is_substitute": "is_sub"})
    roster = roster.assign(round=1, cost=np.nan)
    fixtures = prediction_lineup(scored_player_fixtures(held, 1, roster)).sample(frac=1, random_state=42)
    assert "actual" not in fixtures
    assert model.opponent_scales["tries"] == 0.5
    units = component_units(held.performances)
    assert units["carried_meters"].unique().tolist() == [19, 0]
    assert list(units.columns) == list(POINTS_PER_UNIT)
    totals = units.groupby([recorded.match_id.to_numpy(), recorded.team.to_numpy()]).sum()
    totals.index.names = ["match_id", "team"]
    oracle = model.predict_components(fixtures, observed_team_units=totals)
    pd.testing.assert_frame_equal(oracle.groupby([fixtures.match_id, fixtures.team]).sum(),
                                  totals.astype(float), check_exact=False)
    assert model.predict(fixtures).index.equals(fixtures.index)
    assert np.isfinite(model.predict(fixtures.assign(player_id=fixtures.player_id + 200))).all()
    with pytest.raises(ValueError, match="incomplete team sheets"):
        model.predict(fixtures.iloc[1:])


    team, player = model.explain_component(fixtures, "carried_meters")
    assert team.opponent_multiplier.eq(1).all()
    factors = team.filter(like="_multiplier")
    np.testing.assert_allclose(team.baseline_units * factors.prod(axis=1), team.expected_units)
    np.testing.assert_allclose(player.groupby(["match_id", "team"]).share.sum(), 1)
    np.testing.assert_allclose(player.expected_units, model.predict_components(fixtures).carried_meters)
