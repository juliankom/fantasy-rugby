"""Choose component shrinkage strengths on expanding-window Six Nations validation folds."""

import argparse
from pathlib import Path

import pandas as pd

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.data.fantasy_fixtures import fixture_path, tournament_rounds
from six_nations_fantasy.data.load import DATA_DIR
from six_nations_fantasy.evaluation.selection import prediction_lineup, scored_player_fixtures
from six_nations_fantasy.models import ComponentShrinkageMean
from six_nations_fantasy.score import SCORE_COMPONENTS


def validation_folds(dataset, years, directory):
    for year in years:
        path = fixture_path(directory, year)
        roster = pd.read_parquet(path) if path.exists() else None
        for round_number, schedule in tournament_rounds(dataset.matches, year):
            round_matches = dataset.select_matches(schedule["match_id"])
            fixtures = scored_player_fixtures(round_matches, round_number, roster)
            if not fixtures.empty:
                training = dataset.matches_before(schedule["date"].min())
                yield training, prediction_lineup(fixtures), fixtures["actual"]


def total_mse(folds, k, overrides):
    squared_error = 0.0
    appearances = 0
    for history, fixtures, observed in folds:
        model = ComponentShrinkageMean(k=k, k_by_stat=overrides)
        model.fit(history)
        error = model.predict(fixtures) - observed
        squared_error += error.pow(2).sum(skipna=False)
        appearances += len(error)
    return squared_error / appearances


def tune(folds, strengths):
    shared = {k: total_mse(folds, k, {}) for k in strengths}
    default = min(shared, key=shared.get)
    best_error = shared[default]
    print(f"Best shared k={default:g}, validation MSE={best_error:.4f}")

    # One greedy pass keeps the search small. Each trial scores the full model.
    overrides = {}
    for stat in SCORE_COMPONENTS:
        for k in strengths:
            if k == default:
                continue
            trial = {**overrides, stat: k}
            error = total_mse(folds, default, trial)
            if error < best_error:
                overrides = trial
                best_error = error
    print(f"Component validation MSE={best_error:.4f}")
    return default, overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2025],
                        help="Validation tournaments preceding the squad backtest")
    parser.add_argument("--k", type=float, nargs="+", default=[0, 5, 20, 80])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--test-year", type=int, default=2026)
    args = parser.parse_args()
    if max(args.years) >= args.test_year:
        parser.error("Validation years must precede the held-out test year")
    folds = list(validation_folds(load_dataset(args.data_dir), sorted(set(args.years)), args.data_dir))
    default, overrides = tune(folds, sorted(set(args.k)))
    options = ["--k", f"{default:g}"]
    if overrides:
        options += ["--component-k", *(f"{stat}={k:g}" for stat, k in overrides.items())]
    print(" ".join(options))



if __name__ == "__main__":
    main()
