"""Compare baselines on the 2026 Six Nations."""

from math import log

import pandas as pd
from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.evaluation import evaluate_holdout
from six_nations_fantasy.models import HistoricalMean, RecencyWeightedMean
from six_nations_fantasy.score import simple_score


def main() -> None:
    dataset = load_dataset()
    dates = dataset.matches["date"]
    match_ids = dataset.matches["match_id"]
    start, end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-04-01")
    training = dataset.select_matches(match_ids[dates < start])
    tournament = dataset.select_matches(match_ids[(dates >= start) & (dates < end)])
    tournament = tournament.filter_competitions(["Six Nations"])
    # Match the previous runner's 0.99 decay per week.
    half_life_days = 7 * log(0.5) / log(0.99)
    models = {
        "historical_mean": HistoricalMean(scorer=simple_score),
        "recency_weighted_mean": RecencyWeightedMean(
            scorer=simple_score, half_life_days=half_life_days
        ),
    }

    players = pd.read_parquet("data/processed/players.parquet")
    for name, model in models.items():
        result = evaluate_holdout(model, training, tournament, scorer=simple_score)
        print(
            f"{name}: n={result.n} MAE={result.mae:.3f} "
            f"RMSE={result.rmse:.3f} bias={result.bias:.3f}"
        )

        print(result.predictions.sort_values(by="predicted", ascending=False).head(5).join(players, on="player_id"))

if __name__ == "__main__":
    main()
