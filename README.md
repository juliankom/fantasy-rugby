# Fantasy Rugby Predictor
A project exploring various statistical models to predicting players' performances in (fantasy) six nations. 

# Data
Currently, we have 69782 performances from 3031 unique players across the major
leagues (URC, Top14, Prem) and competitions (Champions Cup, Six Nations). 
Most statistics relevant for calculating fantasy score, such as meters carried, tackles, 
etc. are tracked.

## Loading and splitting data

The authoritative processed dataset snapshots live in `data/processed/`:
`matches.parquet`, `player_performances.parquet`, `players.parquet`, and
`team_performances.parquet`. These are the unchanged files previously stored in
`notebooks/data/`. Historical snapshots previously in `notebooks/old_data/`
are preserved in `data/archive/`; they are **not** used by the default loader.

The exploratory scraping notebooks are unchanged and some still use relative
filenames for intermediate data; their working directories may need adjusting
when rerunning those historical experiments.

Install locally from the repository root:

```bash
python -m pip install -e '.[dev]'
```

Load two Parquet tables (`matches.parquet` and `player_performances.parquet`)
with a validated `match_id` relationship, then partition by match date:

```python
from six_nations_fantasy.data import load_dataset, temporal_split

dataset = load_dataset()  # defaults to data/processed/ at the repository root
# Or: dataset = load_dataset('/path/to/processed/data')

splits = temporal_split(
    dataset,
    validation_start='2025-01-01',
    test_start='2026-01-01',
)

training_rows = splits.train.player_matches()
validation_rows = splits.validation.player_matches()
test_rows = splits.test.player_matches()

# Evaluate on Six Nations only while retaining all competitions for training:
six_nations_test = splits.test.filter_competitions(['Six Nations'])
```

The default location is resolved from the source package's location, not the
current working directory, so it works from notebooks or scripts in the
repository. When using an installed distribution outside the source checkout,
pass an explicit directory to `load_dataset`.

The windows are **train** (before `validation_start`), **validation** (from
`validation_start` to before `test_start`), and **test** (from `test_start`
onward). Cutoffs are inclusive on the right-hand partition. Every performance
from a match follows that match into the same partition, even when several
players have records for that match. Splits do not shuffle or refit anything;
all three partitions may be empty if the selected dates fall outside the data.
Dates must be timezone-naive; missing or duplicate match IDs, missing dates,
and orphaned performances cause clear errors.

`RugbyDataset` preserves the original performance columns. `player_matches()`
adds match metadata (including dates and competition), ready for modelling.
For evaluation, avoid supplying the model with outcome columns from the match
it is trying to predict; chronological splits alone cannot prevent feature
leakage. For now, `players.parquet` and `team_performances.parquet` are left
out of the dataset abstraction until their intended feature semantics are
specified.

Run the new automated data tests with:

```bash
python -m pytest
```

The old `tests/main.py` and `tests/test_selection.py` remain manual experiments
and are deliberately excluded from the automated test discovery. They will be
reorganized in a later cleanup.

# Aims
To explore different statistical models predicting players' fantasy scores in a given
game (best would be specifically for Six Nations games). The baseline
here is the naive predictor taking the mean of all of a players previous performances.

Currently implemented are:
- The naive mean
- The mean weighted to favour recent performances (using exponential decay)
- A Bayesian model assuming that fantasy scores are log-normally distributed, allowing us to calculate a shrinkage factor
    for players with few minutes and few appearances

Future: 
- Cleaner testing pipeline 
- Models made specifically to predict individual stats like tackles 
- Random forest models
- Gradient boosting

### Notes
Make sure to ``.venv/bin/activate``.

## Evaluating models

The evaluation pipeline keeps **historical outcomes** separate from the
**fixture context** passed to a model. It uses the match-safe chronological
splits described above, and reports MAE, RMSE and signed mean error (bias).

A model implements `fit(history)` and `predict(fixtures)`:

- `fit` receives `match_id`, `player_id`, `date`, and `fantasy_points` computed
  from historical performances.
- `predict` receives only `match_id`, `player_id`, and `date`, never the actual
  performance statistics or targets for the match being predicted.
- `predict` must return one finite numeric `pandas.Series` prediction per input
  row, with the same index. Unknown players use the global mean in the supplied
  baseline models.

For a **single tournament**, choose an evaluation date window, train on
all earlier matches, and compare as many model configurations as you like.
The evaluator fits each model on the supplied historical `RugbyDataset` and
returns predictions and error metrics for the evaluation dataset. It does **not**
select a winning model or automatically trigger a second evaluation.

```python
import pandas as pd

from six_nations_fantasy.data import load_dataset, temporal_split
from six_nations_fantasy.evaluation import (
    HistoricalMean, RecencyWeightedMean, evaluate_holdout,
)
from six_nations_fantasy.score import score

dataset = load_dataset()
dates = dataset.matches["date"]
ids = dataset.matches["match_id"]
start, end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-04-01")

training = dataset.select_matches(ids[dates < start])
tournament = dataset.select_matches(ids[(dates >= start) & (dates < end)])
tournament = tournament.filter_competitions(["Six Nations"])

models = {
    "historical_mean": HistoricalMean(scorer=score),
    "weighted_90_days": RecencyWeightedMean(scorer=score, half_life_days=90),
    "weighted_180_days": RecencyWeightedMean(scorer=score, half_life_days=180),
}
for name, model in models.items():
    result = evaluate_holdout(model, training, tournament, scorer=score)
    print(name, result.n, result.mae, result.rmse, result.bias)
```

The date range is **start-inclusive, end-exclusive**. Every model receives the
same training matches and evaluation matches, making their metrics comparable.
The evaluation scorer should use the same rules as any model that trains on
historical fantasy points.

If you want a **train/validation/test** study instead, PR #1's
`temporal_split()` is still available. You decide when to evaluate each period:

```python
splits = temporal_split(dataset, validation_start="2025-01-01", test_start="2026-01-01")
validation_result = evaluate_holdout(
    HistoricalMean(scorer=score), splits.train,
    splits.validation.filter_competitions(["Six Nations"]), scorer=score,
)
# You may later evaluate the final test separately, if your study calls for it.
```

A **holdout** is simply whichever set of completed matches is being used for
an evaluation; it can be a validation set, a test set, or a chosen tournament.
An **empty holdout** has no matches or no player-performance records. A date
window beyond the latest recorded match, or filtering to a competition with
no games in that window, produces an empty holdout. The evaluator raises
`ValueError` instead of returning meaningless MAE/RMSE values. Training must
also contain matches and performances, and every training match must precede
every evaluation match.

Run a tournament-level historical-mean evaluation from the source checkout:

```bash
python scripts/evaluate_baseline.py \
    --evaluation-start 2026-02-01 \
    --evaluation-end 2026-04-01 \
    --competition 'Six Nations'
```

The script trains on all matches before the start date and evaluates on the
chosen window. `--evaluation-end` is optional; omit it to evaluate on all
subsequent matches. The competition filter applies only to evaluation matches.

**Limitations:** The default `score()` function is the project's provisional
fantasy scoring approximation and has not been verified against the current
Parquet dataset in this PR. Replace it with a better scoring callable as needed.
This framework is a **fixed holdout** evaluator, not yet a rolling backtest.
Predictions deliberately have a small input schema; richer pre-match features
(such as announced lineup or opponent) must be explicitly added and validated
rather than copied from post-match performance tables.
