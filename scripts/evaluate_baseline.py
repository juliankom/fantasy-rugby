"""Evaluate the historical-mean baseline over a chosen tournament window.

Run from a source checkout after `pip install -e '.[dev]'`:
    python scripts/evaluate_baseline.py --evaluation-start 2026-02-01 \
        --evaluation-end 2026-04-01 --competition 'Six Nations'

The evaluation end is exclusive. With no end, evaluate all later matches.
Training includes every match strictly before the evaluation start date.
"""

import argparse
from datetime import date

import pandas as pd

from six_nations_fantasy.data import RugbyDataset, load_dataset
from six_nations_fantasy.evaluation import HistoricalMean, evaluate_holdout
from six_nations_fantasy.score import score


def select_evaluation_period(
    dataset: RugbyDataset,
    *,
    start: date,
    end: date | None = None,
    competition: str | None = None,
) -> tuple[RugbyDataset, RugbyDataset]:
    """Select all matches before start for training and [start, end) to evaluate.

    Competition filtering applies only to evaluation, not the training history.
    All player performances follow their match via RugbyDataset.select_matches().
    """
    if end is not None and end <= start:
        raise ValueError("evaluation end must be later than evaluation start")

    matches = dataset.matches
    dates = matches["date"]
    start_ts = pd.Timestamp(start)
    history = dataset.select_matches(matches.loc[dates < start_ts, "match_id"])

    evaluate_mask = dates >= start_ts
    if end is not None:
        evaluate_mask &= dates < pd.Timestamp(end)
    evaluation = dataset.select_matches(matches.loc[evaluate_mask, "match_id"])
    if competition is not None:
        evaluation = evaluation.filter_competitions([competition])
    return history, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-start", required=True, type=date.fromisoformat,
                        help="First date included in evaluation (YYYY-MM-DD)")
    parser.add_argument("--evaluation-end", type=date.fromisoformat,
                        help="First date excluded from evaluation (YYYY-MM-DD)")
    parser.add_argument("--competition", default=None,
                        help="Filter evaluation matches only")
    parser.add_argument("--data", default=None, help="Override data/processed directory")
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    try:
        training, evaluation = select_evaluation_period(
            dataset, start=args.evaluation_start, end=args.evaluation_end,
            competition=args.competition,
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = evaluate_holdout(
        HistoricalMean(scorer=score), training, evaluation, scorer=score
    )
    print(f"n={result.n} MAE={result.mae:.3f} RMSE={result.rmse:.3f} bias={result.bias:.3f}")


if __name__ == "__main__":
    main()
