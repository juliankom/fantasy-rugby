"""Choose component shrinkage strengths on expanding-window Six Nations validation folds."""

import argparse

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.models import ComponentShrinkageMean
from six_nations_fantasy.score import SCORE_COMPONENTS, simple_score


def validation_folds(dataset, years):
    matches = dataset.matches
    tournament = dataset.filter_competitions(["Six Nations"])
    for year in years:
        season = tournament.matches.loc[tournament.matches["date"].dt.year.eq(year)]
        if season.empty:
            raise ValueError(f"No Six Nations matches in {year}")
        history = dataset.select_matches(matches.loc[matches["date"] < season["date"].min(), "match_id"])
        actual = tournament.select_matches(season["match_id"]).player_matches()
        fixtures = actual.filter(items=["player_id", "number", "is_substitute"])
        yield history, fixtures, simple_score(actual)


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
    parser.add_argument("--years", type=int, nargs="+", required=True,
                        help="Validation tournaments preceding the squad backtest")
    parser.add_argument("--k", type=float, nargs="+", default=[0, 5, 20, 80])
    args = parser.parse_args()
    folds = list(validation_folds(load_dataset(), sorted(set(args.years))))
    default, overrides = tune(folds, sorted(set(args.k)))
    options = ["--k", f"{default:g}"]
    if overrides:
        options += ["--component-k", *(f"{stat}={k:g}" for stat, k in overrides.items())]
    print(" ".join(options))



if __name__ == "__main__":
    main()
