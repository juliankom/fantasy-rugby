import os
from copy import deepcopy

import pytest

from six_nations_fantasy.scraping import scrape_match
from six_nations_fantasy.scraping.fixtures import _fixtures
from six_nations_fantasy.scraping.match import minutes_played, parse_match


def test_wrong_game_rejected_before_reading_stats():
    with pytest.raises(ValueError, match="Requested match 947106, received 292034"):
        parse_match('const mcApp = createApp({game:{"id":"292034"}})',
                    "https://www.rugbypass.com/live/england-vs-wales/?g=947106")


def test_multiple_stints_and_stoppage_time():
    changes = [{"minute": minute, "second": 0, "type": event} for minute, event in
               [(81, "sub-on"), (17, "sub-on"), (27, "sub-off")]]
    assert minutes_played(17, changes) == 10
    assert minutes_played(1, []) == 80
    assert minutes_played(16, []) == 0


def test_discovery_includes_all_tournaments_and_finals():
    game = dict(id=1, url="https://www.rugbypass.com/live/a-vs-b/?g=1",
                tournamentUri="six-nations", played=True, hasLink=True,
                homeTeam={"name": "A"}, awayTeam={"name": "B"}, homeScore=10, awayScore=3)
    final = dict(game, id=2, url=game["url"][:-1] + "2")
    regular = {"tournaments": [{"games": []}, {"games": [game]}]}
    payload = {"currentGameDays": [regular], "currentFinalGameDays": [
        {"tournaments": [{"games": [deepcopy(game), final]}]}]}
    season = {"games": 2, "label": "2026"}
    assert [x["match_id"] for x in _fixtures(payload, season, "six-nations")] == [1, 2]
    payload["currentFinalGameDays"] = []
    with pytest.raises(ValueError, match="Incomplete listing"):
        list(_fixtures(payload, season, "six-nations"))


@pytest.mark.skipif(not os.getenv("RUGBYPASS_LIVE"), reason="requires RugbyPass access")
def test_live_england_wales_2026():
    result = scrape_match("https://www.rugbypass.com/live/england-vs-wales/?g=290467")
    assert (result.match["home_score"], result.match["away_score"]) == (48, 7)
    assert result.match["date"].isoformat() == "2026-02-07"
    players = {p["player_identifier"]: p for p in result.performances}
    assert len(players) == 46
    freeman = players["tommy-freeman"]
    assert (freeman["minutes_played"], freeman["tries"], freeman["clean_breaks"]) == (80, 1, 4)
    ford = players["george-ford"]
    assert (ford["penalty_goals"], ford["conversions"]) == (1, 4)
    assert players["alex-mann"]["minutes_played"] == 40
