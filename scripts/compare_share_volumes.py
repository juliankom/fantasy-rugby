"""Compare forecast and actual team volumes using the same fitted player shares."""

import argparse
from math import isfinite
from pathlib import Path

import pandas as pd

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.data.fantasy_fixtures import fixture_path, tournament_rounds
from six_nations_fantasy.data.load import DATA_DIR
from six_nations_fantasy.evaluation.pipeline import recorded_lineup
from six_nations_fantasy.evaluation.selection import (
    prediction_lineup,
    scored_player_fixtures,
    summarize,
)
from six_nations_fantasy.evaluation.squads import select_squad, squad_solver
from six_nations_fantasy.models import PlayerShareModel
from six_nations_fantasy.models.player_share import POINTS_PER_STAT, component_units


def compare_round(training, round_matches, round_number, roster, solver, tau):
    scored = scored_player_fixtures(round_matches, round_number, roster)
    if scored.empty:
        return [], None

    fixtures = prediction_lineup(scored)
    recorded = recorded_lineup(round_matches)
    side_keys = ["match_id", "team"]
    # Use all recorded players for the team totals, including late replacements
    # absent from the announced roster. Allocation still uses the same announced
    # players in both runs.
    actual_counts = component_units(round_matches.performances)
    actual_totals = actual_counts.groupby([recorded[key].to_numpy() for key in side_keys]).sum()
    actual_totals.index.names = side_keys

    model = PlayerShareModel(tau=tau)
    model.fit(training)
    forecast_units = model.predict_components(fixtures)
    diagnostic_units = model.predict_components(fixtures, observed_team_units=actual_totals)
    weights = pd.Series(POINTS_PER_STAT)
    forecasts = {
        "Forecast team volumes": forecast_units.mul(weights).sum(axis=1),
        "Actual team volumes": diagnostic_units.mul(weights).sum(axis=1),
    }

    forecast_totals = forecast_units.groupby([fixtures[key].to_numpy() for key in side_keys]).sum()
    forecast_totals.index.names = side_keys
    volume_errors = forecast_totals.sub(actual_totals.reindex(forecast_totals.index)).abs()

    print(f"Round {round_number} | {len(fixtures)} players | {len(training.matches):,} training matches")
    player_keys = ["match_id", "player_id"]
    announced_ids = pd.MultiIndex.from_frame(fixtures[player_keys])
    recorded_ids = pd.MultiIndex.from_frame(recorded[player_keys])
    if roster is not None:
        print(f"  Announced without a recorded row: {len(announced_ids.difference(recorded_ids))}")

    optimum = None
    if roster is not None:
        candidates = scored.dropna(subset=["cost"])
        optimum = select_squad(candidates, "actual", solver)["actual_total"].sum()

    results = []
    for label, points in forecasts.items():
        squared_error = points.sub(scored["actual"]).pow(2)
        print(f"  {label}: player RMSE {squared_error.mean() ** 0.5:.2f}", end="")
        squad_points = None
        if roster is not None:
            squad = select_squad(candidates.assign(predicted=points), "predicted", solver)
            squad_points = squad["actual_total"].sum()
            captain = squad.loc[squad["captain"], "name"].iloc[0]
            supersub = squad.loc[squad["supersub"], "name"].iloc[0]
            print(f" | {squad_points:.1f} realized points | captain {captain} | supersub {supersub}", end="")
        print()
        results.append(dict(model=label, players=len(points), squared_error=squared_error.sum(),
                            squad_points=squad_points, optimum=optimum))
    return results, volume_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", nargs="?", type=int, default=2026)
    parser.add_argument("--round", type=int, choices=range(1, 6), dest="round_number")
    parser.add_argument("--tau", type=float, default=0.25, help="Opponent scale (0 disables opposition)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    if not isfinite(args.tau) or args.tau < 0:
        parser.error("--tau must be finite and nonnegative")

    dataset = load_dataset(args.data_dir)
    roster_path = fixture_path(args.data_dir, args.year)
    roster = pd.read_parquet(roster_path) if roster_path.exists() else None
    solver = squad_solver() if roster is not None else None
    lineup_source = "announced rosters" if roster is not None else "recorded lineups"
    print(f"{args.year} Six Nations | tau={args.tau:g} | {lineup_source}")
    print("Actual team volumes use post-match data for a retrospective diagnostic.")

    results, volume_errors = [], []
    for round_number, schedule in tournament_rounds(dataset.matches, args.year):
        if args.round_number is not None and args.round_number != round_number:
            continue
        training = dataset.matches_before(schedule["date"].min())
        round_matches = dataset.select_matches(schedule["match_id"])
        round_results, round_errors = compare_round(
            training, round_matches, round_number, roster, solver, args.tau
        )
        if round_results:
            results.extend(round_results)
            volume_errors.append(round_errors)
    if not results:
        parser.error("No player fixtures in the requested rounds")

    print("\nOverall")
    print(summarize(results).to_string(float_format="{:.2f}".format))
    print("\nMean absolute team-volume forecast error by stat (in each stat's original units)")
    print(pd.concat(volume_errors).mean().to_string(float_format="{:.2f}".format))


if __name__ == "__main__":
    main()
