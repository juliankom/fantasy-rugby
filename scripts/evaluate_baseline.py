"""Evaluate a simple historical-mean model on held-out seasons.

Run from a source checkout after `pip install -e '.[dev]'`:
    python scripts/evaluate_baseline.py --validation-start 2025-01-01 --test-start 2026-01-01
"""

import argparse

from six_nations_fantasy.data import load_dataset, temporal_split
from six_nations_fantasy.evaluation import HistoricalMean, evaluate_splits
from six_nations_fantasy.score import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--competition", default=None, help="Filter evaluation matches only")
    parser.add_argument("--data", default=None, help="Override data/processed directory")
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    split = temporal_split(
        dataset,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )
    results = evaluate_splits(
        HistoricalMean, split, scorer=score, competition=args.competition
    )
    for name, result in results.items():
        print(f"{name}: n={result.n} MAE={result.mae:.3f} RMSE={result.rmse:.3f} bias={result.bias:.3f}")


if __name__ == "__main__":
    main()
