"""Validate model settings on earlier tournaments, then test a chosen setting."""

import argparse
from math import log
from pathlib import Path

import pandas as pd

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.data.fantasy_fixtures import fixture_path, tournament_rounds
from six_nations_fantasy.data.load import DATA_DIR
from six_nations_fantasy.evaluation.pipeline import fit_predict
from six_nations_fantasy.evaluation.selection import prediction_lineup, scored_player_fixtures, summarize
from six_nations_fantasy.evaluation.squads import (
    print_squad, select_squad, single_player_exclusions, squad_solver,
)
from six_nations_fantasy.models import (
    ComponentShrinkageMean, HistoricalMean, PositionShrinkageMean, RecencyWeightedMean,
)
from six_nations_fantasy.score import simple_score


VALIDATION_YEARS = [2025]
TEST_YEAR = 2026


def models_to_compare(strengths, component_k):
    models = {
        "Historical mean": HistoricalMean(scorer=simple_score),
        "Recency-weighted mean": RecencyWeightedMean(
            scorer=simple_score, half_life_days=7 * log(0.5) / log(0.85)
        ),
    }
    for k in strengths:
        models[f"Position shrinkage (k={k:g})"] = PositionShrinkageMean(scorer=simple_score, k=k)
        if component_k:
            models[f"Component shrinkage (k={k:g}, {component_k})"] = ComponentShrinkageMean(
                k=k, k_by_stat=component_k
            )
    return models


def score_squad(scored_fixtures, objective, solver):
    squad = select_squad(scored_fixtures.dropna(subset=["cost"]), objective, solver)
    points = squad["actual_total"].sum()
    print(f"Cost {squad['cost'].sum():.1f}, objective {squad['objective_total'].sum():.1f}, actual {points:.1f}")
    print_squad(squad)
    return squad, points


def evaluate_on_six_nations(dataset, year, models, fantasy_roster, *, single_player_drops=False):
    solver = squad_solver()
    results = []
    for round_number, schedule in tournament_rounds(dataset.matches, year):
        round_matches = dataset.select_matches(schedule["match_id"])
        scored_fixtures = scored_player_fixtures(round_matches, round_number, fantasy_roster)
        if scored_fixtures.empty:  # A historical roster may cover only some rounds.
            continue

        round_start = schedule["date"].min()
        training = dataset.matches_before(round_start)
        fixtures = prediction_lineup(scored_fixtures)
        print(f"\n{year} Six Nations round {round_number} | training on {len(training.matches)} "
              f"matches from all competitions before {round_start:%Y-%m-%d}")
        optimum = None
        if fantasy_roster is None:
            print("Recorded lineups · player RMSE only")
        else:
            print("Hindsight optimum")
            best, optimum = score_squad(scored_fixtures, "actual", solver)
            if single_player_drops:
                print(single_player_exclusions(scored_fixtures.dropna(subset=["cost"]), best, solver).to_string(index=False))

        for label, model in models.items():
            predicted = fit_predict(model, training, fixtures)
            squared = (predicted - scored_fixtures["actual"]).pow(2)
            print(f"{label} | player RMSE {squared.mean() ** 0.5:.2f}")
            points = None
            if fantasy_roster is not None:
                _, points = score_squad(scored_fixtures.assign(predicted=predicted), "predicted", solver)
            results.append(dict(round=round_number, model=label, players=len(fixtures),
                                squared_error=squared.sum(), squad_points=points, optimum=optimum))
    print_tournament_results(year, results)


def print_tournament_results(year, results):
    print(summarize(results).to_string(float_format="{:.2f}".format))
    scores = pd.DataFrame(results).pivot(index="round", columns="model", values="squad_points")
    if scores.notna().any().any():
        scores["Hindsight optimum"] = pd.DataFrame(results).groupby("round")["optimum"].first()
        frg_path = Path(__file__).with_name("frg_simple_scores.parquet")
        if year == 2026 and frg_path.exists():
            scores["FRG score"] = pd.read_parquet(frg_path).set_index("round_number")["score"]
        scores.loc["total"] = scores.sum()
        print("\nRealized squad points")
        print(scores.to_string(float_format="{:.1f}".format))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validate", "test"], nargs="?", default="validate")
    parser.add_argument("--validation-years", nargs="+", type=int, default=VALIDATION_YEARS)
    parser.add_argument("--test-year", type=int, default=TEST_YEAR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--k", nargs="+", type=float, default=[5.0])
    parser.add_argument("--component-k", nargs="+", metavar="STAT=K", default=[])
    parser.add_argument("--single-player-drops", action="store_true")
    args = parser.parse_args()
    if args.stage == "validate" and max(args.validation_years) >= args.test_year:
        parser.error("Validation years must precede the held-out test year")
    overrides = {}
    for setting in args.component_k:
        stat, k = setting.split("=")
        overrides[stat] = float(k)
    models = models_to_compare(args.k, overrides)
    dataset = load_dataset(args.data_dir)
    years = args.validation_years if args.stage == "validate" else [args.test_year]
    for year in years:
        roster_path = fixture_path(args.data_dir, year)
        if args.stage == "test" and not roster_path.exists():
            parser.error(f"Squad testing requires a prepared fantasy roster: {roster_path}")
        fantasy_roster = pd.read_parquet(roster_path) if roster_path.exists() else None
        evaluate_on_six_nations(dataset, year, models, fantasy_roster,
                                single_player_drops=args.single_player_drops)


if __name__ == "__main__":
    main()
