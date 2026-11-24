# Fantasy Rugby Predictor
A project exploring various statistical models to predicting players' performances in (fantasy) six nations. 

# Data

## Loading and splitting data

Place your local dataset in `data/rebuild/` or pass its directory to
`load_dataset()`. The loader requires `matches.parquet` and
`player_performances.parquet`. The squad backtest also uses `players.parquet`.
Parquet datasets, archives and fixture manifests are ignored by Git and must
be supplied separately in a fresh checkout.

Install locally from the repository root:

```bash
python -m pip install -e '.[dev]'
```

Load two Parquet tables (`matches.parquet` and `player_performances.parquet`)
with a validated `match_id` relationship, then partition by match date:

```python
from six_nations_fantasy.data import load_dataset, temporal_split

dataset = load_dataset()  # defaults to data/rebuild/ at the repository root
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

Run `python tests/main.py` from the source checkout to compare the historical
mean and recency-weighted mean on the 2026 Six Nations. The runner loads
`data/rebuild/` by default and prints MAE, RMSE, bias, and the highest-rated players for both models.
### Validate and test squad selection

From an editable install, run validation first. The default validation year is
2025. Before each round, refit each model on every available match from every competition
dated before the round's opening fixture. For example, training for 2026 round 2
includes earlier seasons, round 1, and any club matches played before round 2.
Only the evaluated fixtures are restricted to that year's Six Nations. Keep the hyperparameters
chosen on validation fixed throughout the test tournament. Compare model definitions in `tests/test_selection.py`.

```bash
python tests/test_selection.py validate --k 5 15 20
```

Validation without a fantasy roster reports player RMSE on recorded lineups.
To also score squads, prepare each year's Google Sheets CSV export once.
The generator expects the same columns and position labels as the bundled
2026 CSV. Pass the year and CSV path, for example:

```bash
python scripts/prepare_fixtures.py 2026 tests/six_nations_2026_fantasy_prices.csv
```

Use the same command with 2024 or 2025 and the corresponding export. Supply
whole rounds for comparable squad scores. The generator writes
`data/rebuild/six_nations_<year>_fixtures.parquet` with player IDs, match IDs,
prices, fantasy positions and announced roles. If the export covers only
some rounds, evaluation uses those rounds.

After choosing hyperparameters, run the held-out tournament explicitly:

```bash
python tests/test_selection.py test --k 15
```

Testing defaults to 2026 and requires its prepared roster. Player RMSE pools
squared errors across all player fixtures in each evaluated season, including
unpriced players. Squad totals use priced players, captain and supersub
multipliers, and project-calculated actual points. The percentage compares
total selected points to the sum of the round-by-round hindsight optima.

#### Player-share model

Compare a few opponent settings on validation rounds:

```bash
python tests/test_selection.py validate --share-tau 0 0.125 0.25 0.5 --k 20
```

Each number runs a separate model. Zero leaves out the opponent effect. The fit
uses the requested scale for every stat. In Python, pass
`PlayerShareModel(tau=0.25, tau_by_stat={"tries": 0.125})` to override a particular
stat. Choose scales on validation rounds before using them for test rounds.

The model makes a separate forecast for each scoring stat. It first sums the
players' stats for each team in each historical match. It fits a Poisson
regression to those totals. For an England home match against Wales, for example,
the predicted total is the historical average times four multipliers: Six
Nations, home, England as the team, and Wales as the opponent.

The regression fits the logarithms of those multipliers. It pulls the logs
toward zero, which pulls the multipliers toward one. If a log multiplier is
`b`, the penalty is `b² / (2 × scale²)`. Halving the opponent scale `tau`
therefore quadruples that penalty. The hand-set scales for competition, home/away
and team are 2, 0.5 and 0.5. An opponent with no training history gets a
multiplier of one.

A second regression estimates who gets credit for the team's total. Each player
gets a weight for their shirt position and another for their own playing history.
Multiply these weights and divide by the sum of the 23 combined weights to get
the player's share. The fit gives more weight to matches with more events for
that stat. An unseen player gets their position weight alone. All bench shirts
use the same position weight. The player multiplier carries across club and
country.

Multiply the share by the predicted team total, then apply the fantasy points
for that stat. This model predicts raw carried metres at 0.1 points per metre.
Negative recorded metres count as zero in its training. The backtest still
scores actual performances with `simple_score()`, which floors each player's
metres in blocks of ten.

Use all 23 players per side, including unpriced players, to calculate shares.
The fit warns about incomplete historical sheets and skips those sides. Prediction
requires a complete sheet. Use [the inspection notebook](notebooks/player_share_weights.ipynb)
to see the fitted team multipliers, player shares and how much they change
forecasts against the recency-weighted mean.

To test how much of the error comes from forecasting team volumes, keep the
fitted player shares and substitute each side's recorded totals:

```bash
python scripts/compare_share_volumes.py 2026 --tau 0.25
```

Add `--round 1` to inspect one round. The script refits before each round and
compares both versions on the same lineups. It reports player RMSE and, when
a prepared roster exists, realized squad points and percentage of the hindsight
optimum. Recorded totals include late replacements. When using announced
rosters, the model still divides those totals among the announced players.
Actual volumes are only available after the match, so treat this comparison
as a diagnostic of the two parts of the model.

Thanks to Reddit user **u/thekai1992** for creating the per-round fantasy
spreadsheets used in the squad backtest.

# Aims
To explore different statistical models predicting players' fantasy scores in a given
game (best would be specifically for Six Nations games). The baseline
here is the naive predictor taking the mean of all of a players previous performances.

Currently implemented are:
- The naive mean
- The mean weighted to favour recent performances (using exponential decay)

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

- `fit` receives a historical `RugbyDataset`. Models that learn from fantasy
  points calculate them during fitting.
- `predict` receives player and match identifiers, date, and available lineup and
  schedule fields (shirt, bench status, team, opponent, competition and home side).
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
from six_nations_fantasy.evaluation import evaluate_holdout
from six_nations_fantasy.models import HistoricalMean, RecencyWeightedMean
from six_nations_fantasy.score import simple_score

dataset = load_dataset()
dates = dataset.matches["date"]
ids = dataset.matches["match_id"]
start, end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-04-01")

training = dataset.select_matches(ids[dates < start])
tournament = dataset.select_matches(ids[(dates >= start) & (dates < end)])
tournament = tournament.filter_competitions(["Six Nations"])

models = {
    "historical_mean": HistoricalMean(scorer=simple_score),
    "weighted_90_days": RecencyWeightedMean(scorer=simple_score, half_life_days=90),
    "weighted_180_days": RecencyWeightedMean(scorer=simple_score, half_life_days=180),
}
for name, model in models.items():
    result = evaluate_holdout(model, training, tournament, scorer=simple_score)
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
    HistoricalMean(scorer=simple_score), splits.train,
    splits.validation.filter_competitions(["Six Nations"]), scorer=simple_score,
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

**Limitations:** The `simple_score()` function is the project's provisional
fantasy scoring approximation. Its outputs can differ from official fantasy
points.
This framework is a **fixed holdout** evaluator, not yet a rolling backtest.
Predictions deliberately have a small input schema; richer pre-match features
(such as announced lineup or opponent) must be explicitly added and validated
rather than copied from post-match performance tables.

## Scraping RugbyPass

### RugbyPass terms and liability

Section 9 of the RugbyPass Terms of Use prohibits scraping RugbyPass without
its prior express written permission. Users are responsible for obtaining
that permission and complying with applicable terms and laws.
[RugbyPass Terms of Use](https://www.rugbypass.com/terms-of-use/)

The project authors disclaim responsibility and liability arising from use
of this code to the extent permitted by applicable law.

### Pipeline usage

The Python pipeline reads the fixture lists and the JSON embedded in completed match pages.
The historical notebooks in `notebooks/scraping/` document the previous site format.
The default discovery scope covers Six Nations 2022–2026 and URC, Top 14,
Premiership and Champions Cup 2022/23–2025/26. Each fixture row identifies a
match, season, competition, opponents and final score.

With RugbyPass's permission, run the full pipeline from the repository root:

```bash
python -m six_nations_fantasy.scraping
```

This refreshes `data/scraping/urls.parquet`, prints fixture counts by season
and competition, and scrapes those URLs into a new `data/rebuild/` directory.
To reuse the saved URLs and choose a fresh output directory:

```bash
python -m six_nations_fantasy.scraping --reuse-manifest --output data/rebuild-2
```

Use `--manifest PATH` to select another URL table. The command exits with
status 1 if any fixtures were rejected. The batch checks
the match page against the manifest, writes the four dataset Parquets for accepted
matches and records rejected fixtures in `failures.parquet`. Inspect failures before
using the rebuild. If every fixture fails, the output contains only the failure report.

The rebuild uses RugbyPass numeric player IDs and profile slugs. The historical
registry assigned its own integers, so load the rebuilt tables together and use
`player_identifier` when comparing registries. Minutes sum substitution stints
with the historical 80-minute cap. Kicking totals count successful lineup events
and are checked against the team totals. With RugbyPass's permission, run the
single-match regression and display the match details, both teams' statistics
and every player performance with:

```bash
RUGBYPASS_LIVE=1 python -m pytest -s -q tests/unit/test_rugbypass.py -k live_england_wales
```
