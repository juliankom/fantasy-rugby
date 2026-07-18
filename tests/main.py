import pandas as pd
import datetime as dt
from six_nations_fantasy.score import score
from six_nations_fantasy.models.naive_mean import *
from six_nations_fantasy.models.weighted_mean import *
from six_nations_fantasy.data.read_json import database_from_dir

# TODO: we need to expand the interface for estimators to include the next
# game information, which is probably VERY important for more involved estimators
matches = pd.read_parquet('all_data_matches.parquet')
perfs = pd.read_parquet('all_data_performances.parquet')

merged = perfs.join(matches, on='match_id')

# for now, we train the model on all matches before 2026
training_data = merged[merged['date'].apply(lambda d: d.year) < 2026]

test_data = merged[merged['date'].apply(lambda d: d.year) == 2026]

# weighted mean (exponential weighting based on weeks since the end of 2025 tournament)
model = WeightedMean(score, training_data, dt.date(2025, 3, 15), decay_factor=0.99)

# mean squared error of predictions compared to actual scores
def test_predictions(model, test_data):
    test_data['score'] = test_data.apply(score, axis=1)
    test_data['prediction'] = test_data.apply(lambda p: model.estimate(p['name']), axis=1)

    test_data.loc[:, 'sq_error'] = (test_data['score'] - test_data['prediction'])**2
    test_data.loc[:, 'abs_error'] = abs(test_data['score'] - test_data['prediction'])

    # compute R^2 
    residual_sum_sq = test_data['sq_error'].sum()
    total_sum_sq = ((test_data['score'] - test_data['score'].mean())**2).sum()
    R_sq = 1 - residual_sum_sq / total_sum_sq

    return test_data[['abs_error', 'sq_error']], R_sq

# TODO at some point, probably should transition to using scikit-learn for 
# this and many other things for ease of use
result_statistics, R_sq = test_predictions(model, test_data)
print(result_statistics.describe())
print(f"R^2 = {R_sq}")
