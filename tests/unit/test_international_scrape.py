from six_nations_fantasy.scraping.international import _structured_results


def test_international_discovery_filters_mixed_and_incomplete_games():
    games = {}
    for identifier, home, away, slug, played in (
        (1, "New Zealand", "France", "nations-championship", True),
        (2, "Belgium", "Romania", "internationals", True),
        (3, "New Zealand", "South Africa", "internationals", True),
        (4, "England", "Barbarians", "internationals", True),
        (5, "Wales", "Japan", "internationals", False),
        (6, "Ireland", "France", "six-nations", True),
    ):
        games[str(identifier)] = {
            "id": identifier, "homeTeam": {"name": home}, "awayTeam": {"name": away},
            "tournamentUri": slug, "played": played, "hasLink": True,
            "dateId": "20260704", "homeScore": 24, "awayScore": 17,
            "url": f"https://www.rugbypass.com/live/match/?g={identifier}",
        }
    games["7"] = {"id": 7, "upcoming": True}
    payload = {
        "currentGameDays": {"0": {"tournaments": {"0": {"games": games}}}},
        "currentFinalGameDays": [],
    }
    rows = list(_structured_results(payload, 2026))
    assert [row["match_id"] for row in rows] == [1, 6]
    assert rows[0]["competition"] == "Nations Championship"
