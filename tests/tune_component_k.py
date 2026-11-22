"""Choose component shrinkage strengths on expanding-window Six Nations validation folds."""

import argparse

import pandas as pd

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.models import PositionShrinkageMean
from six_nations_fantasy.score import SCORE_COMPONENTS


def validation_errors(dataset, years, strengths):
    matches = dataset.matches
    tournament = dataset.filter_competitions(["Six Nations"])
    results = []
    for year in years:
        season = tournament.matches.loc[tournament.matches["date"].dt.year.eq(year)]
        if season.empty:
            raise ValueError(f"No Six Nations matches in {year}")
        history = dataset.select_matches(matches.loc[matches["date"] < season["date"].min(), "match_id"])
        actual = tournament.select_matches(season["match_id"]).player_matches()
        fixtures = actual.filter(items=["player_id", "number", "is_substitute"])
        for stat, scorer in SCORE_COMPONENTS.items():
            observed = scorer(actual)
            for k in strengths:
                model = PositionShrinkageMean(scorer=scorer, k=k)
                model.fit(history)
                errors = (observed - model.predict(fixtures)).pow(2)
                results.append((stat, k, errors.sum(), errors.count()))
    return pd.DataFrame(results, columns=["stat", "k", "squared_error", "appearances"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", required=True,
                        help="Validation tournaments preceding the squad backtest")
    parser.add_argument("--k", type=float, nargs="+", default=[0, 5, 20, 80])
    args = parser.parse_args()
    errors = validation_errors(load_dataset(), sorted(set(args.years)), args.k)
    totals = errors.groupby(["stat", "k"])[["squared_error", "appearances"]].sum()
    mse = (totals["squared_error"] / totals["appearances"]).unstack("k")
    print(mse)
    best = mse.idxmin(axis=1)
    print("--component-k", *(f"{stat}={k:g}" for stat, k in best.items()))


if __name__ == "__main__":
    main()
