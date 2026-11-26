import pandas as pd

from six_nations_fantasy.evaluation.selection import compare_price_free_picks, price_free_picks


def test_price_free_ranking_and_changes():
    # Two sides, with deliberately different realized scores for tied candidates.
    shirts = list(range(1, 24)) * 2
    fixtures = pd.DataFrame({
        "player_id": range(1, 47), "match_id": 1, "number": shirts,
        "is_substitute": [number > 15 for number in shirts], "actual": range(1, 47),
    }).sample(frac=1, random_state=1)
    predictions = pd.Series(0., index=fixtures.index)
    baseline = price_free_picks(fixtures, predictions)
    assert sorted(baseline.player_id) == list(range(1, 17))
    predictions.loc[fixtures.player_id.isin([25, 46])] = 10
    picks = price_free_picks(fixtures, predictions)
    comparison = compare_price_free_picks(picks, baseline)
    assert comparison["XV changes"] == comparison["bench changes"] == 1
    assert comparison["XV gain"] == 25 - 2
    assert comparison["bench gain"] == 46 - 16
    assert len(picks) == 16


def test_subset_scores_share_pool_and_preserve_positional_slots():
    import numpy as np
    from six_nations_fantasy.evaluation.validation import subset_scores

    shirts = list(range(1, 24)) * 2
    fixtures = pd.DataFrame({
        "number": shirts, "player_id": range(46), "match_id": 1,
        "is_substitute": [shirt > 15 for shirt in shirts],
        "actual": [100 if shirt > 15 else 1 for shirt in shirts],
    })
    forecasts = pd.Series(range(46))
    predictions = {"baseline": forecasts, "same ranking": forecasts + 5}
    scores = subset_scores(fixtures, predictions, np.random.default_rng(42))
    assert scores.eq(15).all().all()
    fixtures["actual"] = forecasts
    scores = subset_scores(fixtures, predictions, np.random.default_rng(42))
    assert scores.baseline.nunique() > 1
    assert scores.baseline.equals(scores["same ranking"])
