"""Compare pre-match try forecasts on complete Six Nations match sheets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.data.fantasy_fixtures import tournament_rounds
from six_nations_fantasy.data.load import DATA_DIR
from six_nations_fantasy.evaluation.pipeline import recorded_lineup
from six_nations_fantasy.models.try_forecasts import TryForecasts


def evaluate_year(dataset, year, model):
    results = []
    for round_number, schedule in tournament_rounds(dataset.matches, year):
        model.fit(dataset.matches_before(schedule["date"].min()))
        holdout = dataset.select_matches(schedule["match_id"])
        lineup = recorded_lineup(holdout)
        shares, team_totals = model.predict_shares(lineup)
        actual = holdout.performances["tries"].to_numpy()
        for name, share in shares.items():
            expected = share * team_totals
            # A scorer assigned zero probability gives infinite log loss.
            # Show those misses separately as well as a common smoothed score.
            # A 1% uniform share gives every shirt a small chance in every model.
            smoothed = 0.99 * share + 0.01 / 23
            results.append(pd.DataFrame({
                "round": round_number, "model": name, "bench": lineup["is_substitute"],
                "tries": actual, "expected": expected, "squared_error": (actual - expected) ** 2,
                "log_loss": -actual * np.log(smoothed),
                "zero_share_tries": np.where(share.eq(0), actual, 0),
            }))
    return pd.concat(results, ignore_index=True)


def summarize(rows, group):
    totals = rows.groupby(group, sort=False).agg(
        players=("tries", "size"), tries=("tries", "sum"), expected=("expected", "sum"),
        squared_error=("squared_error", "sum"), loss=("log_loss", "sum"),
        zero_share_tries=("zero_share_tries", "sum"),
    )
    totals["loss_per_try"] = totals.pop("loss") / totals["tries"].replace(0, np.nan)
    totals["count_RMSE"] = (totals.pop("squared_error") / totals["players"]) ** 0.5
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prior-opportunities", type=float, default=50)
    parser.add_argument("--half-life-days", type=float, help="Weight older shirt and player records less")
    args = parser.parse_args()
    model = TryForecasts(prior_opportunities=args.prior_opportunities, half_life_days=args.half_life_days)
    rows = evaluate_year(load_dataset(args.data_dir), args.year, model)
    print(f"{args.year} Six Nations | lower loss per try and count RMSE are better")
    print(summarize(rows, "model").round(3).to_string())
    print("\nLoss per credited try by round (1% uniform smoothing)")
    print(summarize(rows, ["round", "model"])["loss_per_try"].unstack().round(3).to_string())
    print("\nStarts and bench appearances")
    print(summarize(rows, ["bench", "model"]).round(3).to_string())


if __name__ == "__main__":
    main()
