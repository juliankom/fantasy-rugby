"""Lineup status from historical match sheets."""

import pandas as pd


def substitute_status(performances: pd.DataFrame) -> pd.Series | None:
    """Read the recorded bench flag or derive it from the lineup number."""
    if "is_substitute" in performances:
        return performances["is_substitute"].astype("boolean")
    if "number" in performances:
        numbers = pd.to_numeric(performances["number"], errors="coerce")
        return numbers.gt(15).astype("boolean").where(numbers.ge(1))
    return None


def match_context(lineup, matches):
    """Attach only scheduled opposition, listed home side and competition."""
    columns = ["match_id", "home_team", "away_team", "competition"]
    if not set(columns).issubset(matches) or not {"home", "team"}.intersection(lineup):
        return lineup
    schedule = matches.set_index("match_id").reindex(lineup["match_id"])
    schedule.index = lineup.index
    if "team" in lineup:
        home = lineup["team"].eq(schedule["home_team"])
        if not (home | lineup["team"].eq(schedule["away_team"])).all():
            raise ValueError("Lineup team does not match scheduled fixture")
    else:
        home = lineup["home"]
        if not home.isin([True, False]).all():
            raise ValueError("Missing or invalid home side")
        home = home.astype(bool)
    return lineup.assign(home=home, competition=schedule["competition"],
                         team=schedule["home_team"].where(home, schedule["away_team"]),
                         opponent=schedule["away_team"].where(home, schedule["home_team"]))
