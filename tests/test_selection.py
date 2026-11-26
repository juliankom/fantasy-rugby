"""Validate model settings on earlier tournaments, then test a chosen setting."""

import argparse
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.data.fantasy_fixtures import fixture_path, tournament_rounds
from six_nations_fantasy.data.load import DATA_DIR
from six_nations_fantasy.evaluation.pipeline import fit_predict
from six_nations_fantasy.evaluation.selection import (
    compare_price_free_picks, prediction_lineup, price_free_picks, scored_player_fixtures, summarize,
)
from six_nations_fantasy.evaluation.validation import plot_calibration, subset_scores
from six_nations_fantasy.evaluation.squads import (
    print_squad, select_squad, single_player_exclusions, squad_solver,
)
from six_nations_fantasy.models import (
    ComponentShrinkageMean, HistoricalMean, PlayerShareModel, PositionShrinkageMean, RecencyWeightedMean,
)
from six_nations_fantasy.score import simple_score


VALIDATION_YEARS = [2023, 2024, 2025]
BASELINE = "Historical mean"
TEST_YEAR = 2026


def models_to_compare(strengths, component_k, share_taus=()):
    models = {
        BASELINE: HistoricalMean(scorer=simple_score),
        "Recency-weighted mean": RecencyWeightedMean(
            scorer=simple_score, half_life_days=180
        ),
    }
    for k in strengths:
        models[f"Position shrinkage (k={k:g})"] = PositionShrinkageMean(scorer=simple_score, k=k)
        if component_k:
            models[f"Component shrinkage (k={k:g}, {component_k})"] = ComponentShrinkageMean(
                k=k, k_by_stat=component_k
            )
    for tau in share_taus:
        models[f"Player shares (tau={tau:g})"] = PlayerShareModel(tau=tau, half_life_days=180)
    return models


def score_squad(scored_fixtures, objective, solver):
    squad = select_squad(scored_fixtures.dropna(subset=["cost"]), objective, solver)
    points = squad["actual_total"].sum()
    print(f"Cost {squad['cost'].sum():.1f}, objective {squad['objective_total'].sum():.1f}, actual {points:.1f}")
    print_squad(squad)
    return squad, points


def evaluate_on_six_nations(dataset, year, models, fantasy_roster, *, single_player_drops=False):
    solver = squad_solver() if fantasy_roster is not None else None
    comparisons = []
    subset_rounds = []
    selected_starters = []
    results = []
    for round_number, schedule in tournament_rounds(dataset.matches, year, price_free=fantasy_roster is None):
        round_matches = dataset.select_matches(schedule["match_id"])
        scored_fixtures = scored_player_fixtures(round_matches, round_number, fantasy_roster)
        if scored_fixtures.empty:  # A historical roster may cover only some rounds.
            continue

        round_start = schedule["date"].min()
        training = dataset.matches_before(round_start)
        fixtures = prediction_lineup(scored_fixtures)
        print(f"\n{year} Six Nations round {round_number} | training on {len(training.matches)} "
              f"matches from all competitions before {round_start:%Y-%m-%d}")
        picks = {}
        forecasts = {}
        optimum = None
        if fantasy_roster is None:
            print("Price-free selection from recorded lineups")
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
            else:
                forecasts[label] = predicted
                picks[label] = price_free_picks(scored_fixtures, predicted)
            results.append(dict(round=round_number, model=label, players=len(fixtures),
                                squared_error=squared.sum(), squad_points=points, optimum=optimum))
        if picks:
            rng = np.random.default_rng([year, round_number])
            subset_rounds.append(subset_scores(scored_fixtures, forecasts, rng))
        for label, selected in picks.items():
            selected_starters.append(selected.loc[selected.position.ne("Bench"),
                                                 ["position", "predicted", "actual"]].assign(model=label))
            comparison = compare_price_free_picks(selected, picks[BASELINE])
            comparisons.append(dict(year=year, round=round_number, model=label, **comparison))
    if results:
        print_tournament_results(year, results)
    return comparisons, subset_rounds, selected_starters


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
    parser.add_argument("--share-tau", nargs="+", type=float, default=[], metavar="SCALE",
                        help="Compare player-share models at these opponent scales (0 disables opposition)")
    parser.add_argument("--single-player-drops", action="store_true")
    parser.add_argument("--calibration-plot", type=Path, default=Path("validation_calibration.png"))
    args = parser.parse_args()
    if args.stage == "validate" and max(args.validation_years) >= args.test_year:
        parser.error("Validation years must precede the held-out test year")
    if any(not isfinite(tau) or tau < 0 for tau in args.share_tau):
        parser.error("--share-tau expects finite, nonnegative scales")
    overrides = {}
    for setting in args.component_k:
        stat, k = setting.split("=")
        overrides[stat] = float(k)
    models = models_to_compare(args.k, overrides, args.share_tau)
    dataset = load_dataset(args.data_dir)
    years = args.validation_years if args.stage == "validate" else [args.test_year]
    comparisons = []
    subset_rounds = []
    selected_starters = []
    for year in years:
        roster_path = fixture_path(args.data_dir, year)
        if args.stage == "test" and not roster_path.exists():
            parser.error(f"Squad testing requires a prepared fantasy roster: {roster_path}")
        fantasy_roster = pd.read_parquet(roster_path) if args.stage == "test" else None
        rows, subsets, starters = evaluate_on_six_nations(
            dataset, year, models, fantasy_roster, single_player_drops=args.single_player_drops
        )
        comparisons.extend(rows)
        subset_rounds.extend(subsets)
        selected_starters.extend(starters)
    if comparisons:
        rounds = pd.DataFrame(comparisons).set_index(["year", "round", "model"])
        print("\nPrice-free selection by round (gain and changed picks versus historical mean)")
        print(rounds.to_string())
        print("\nPrice-free totals across all validation rounds")
        print(rounds.groupby(level="model", sort=False).sum().to_string())
        scores = sum(subset_rounds)
        gains = scores.subtract(scores[BASELINE], axis="index")
        summary = gains.quantile([.025, .5, .975]).T
        summary.columns = ["2.5%", "median", "97.5%"]
        summary["% wins"] = 100 * gains.gt(0).mean()
        summary["% ties"] = 100 * gains.eq(0).mean()
        print("\nShared player subsets: total XV gain versus historical mean")
        print(summary.to_string(float_format="{:.1f}".format))
        plot_calibration(pd.concat(selected_starters), args.calibration_plot)
        print(f"Selected XV calibration: {args.calibration_plot}")


if __name__ == "__main__":
    main()
