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
