"""Resolve one Six Nations spreadsheet roster to player and match IDs."""

import argparse
from pathlib import Path

import pandas as pd

from six_nations_fantasy.data.roster_names import add_name_keys, name_key
from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.data.load import DATA_DIR
from six_nations_fantasy.data.fantasy_fixtures import fixture_path, tournament_rounds

POSITIONS = {
    "hooker": "Hooker", "prop": "Prop", "second_row": "Second-row",
    "back_row": "Back-row", "scrum_half": "Scrum-half", "fly_half": "Fly-half",
    "centre": "Centre", "back_three": "Back Three",
}


def resolve_round(roster, matches, performances, players):
    teams = pd.concat([
        matches[["match_id", "date", side]].rename(columns={side: "team"})
        for side in ["home_team", "away_team"]
    ])
    resolved_roster = roster.merge(teams, on="team", validate="many_to_one", how="left")
    if resolved_roster["match_id"].isna().any():
        raise ValueError("Roster team has no scheduled match in this round")

    appearances = performances.merge(players, on="player_id", validate="many_to_one")
    appearances["team"] = appearances["home_team"].where(appearances["home"], appearances["away_team"])
    keys = ["match_id", "team", "name_key"]
    resolved_roster = resolved_roster.merge(add_name_keys(appearances)[keys + ["player_id"]], on=keys,
                      how="left", validate="one_to_one")

    # Late withdrawals still need their historical ID and their announced role.
    missing = resolved_roster["player_id"].isna()
    names = players.assign(name_key=players["name"].map(name_key))
    for index, row in resolved_roster.loc[missing].iterrows():
        ids = names.loc[names["name_key"].eq(row["name_key"]), "player_id"]
        if len(ids) != 1:
            raise ValueError(f"Could not resolve {row['team']} player {row['name']}")
        resolved_roster.at[index, "player_id"] = ids.iloc[0]
    resolved_roster["player_id"] = resolved_roster["player_id"].astype(players["player_id"].dtype)
    return resolved_roster


def prepare_fixtures(csv, year, dataset, players):
    roster = pd.read_csv(csv).rename(columns={"player": "name", "price": "cost"})
    roster = add_name_keys(roster.loc[roster["year"].eq(year)].copy())
    if roster.empty or not roster["round"].isin(range(1, 6)).all():
        raise ValueError(f"Expected roster rows for rounds 1–5 of {year}")
    roster["nominal_pos"] = roster["nominal_position"].map(POSITIONS)
    if roster["nominal_pos"].isna().any() or not roster["selection"].isin(["starter", "bench"]).all():
        raise ValueError("Unknown fantasy position or selection")
    roster["is_sub"] = roster["selection"].eq("bench")
    columns = ["year", "round", "name", "name_key", "team", "cost", "nominal_pos", "shirt_number", "is_sub"]
    rounds = []
    for round_number, matches in tournament_rounds(dataset.matches, year):
        entries = roster.loc[roster["round"].eq(round_number), columns]
        if not entries.empty:
            performances = dataset.select_matches(matches["match_id"]).player_matches()
            rounds.append(resolve_round(entries, matches, performances, players))
    return pd.concat(rounds, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", type=int)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    players = pd.read_parquet(args.data_dir / "players.parquet")
    if "player_id" not in players:
        players = players.reset_index()
    fixtures = prepare_fixtures(args.csv, args.year, load_dataset(args.data_dir), players[["player_id", "name"]])
    path = fixture_path(args.data_dir, args.year)
    fixtures.to_parquet(path, index=False)
    print(f"Saved {len(fixtures)} player fixtures to {path}")


if __name__ == "__main__":
    main()
