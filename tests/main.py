import pandas as pd
import datetime as dt
from six_nations_fantasy.score import score
from six_nations_fantasy.models.naive_mean import *
from six_nations_fantasy.models.weighted_mean import *
from six_nations_fantasy.data.read_json import database_from_dir

# TODO: 
# - write entire database somewhere to a single file
# - create separate table for matches 
training_data = database_from_dir('training_data')

# TODO: we need to expand the interface for estimators to include the next
# game information, which is probably VERY important for more involved estimators

# weighted mean (exponential weighting based on weeks since the end of 2025 tournament)
naive = WeightedMean(score, training_data, dt.date(2025, 3, 15), decay_factor=0.99)

# LBB esimated score
#print(naive.estimate('Louis Bielle-Biarrey'))

# Top 10 performers according to this estimator
#print(naive.estimates.sort_values().tail(20))

test_data = database_from_dir('test_data')

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
result_statistics, R_sq = test_predictions(naive, test_data)
print(result_statistics.describe())
print(f"R^2 = {R_sq}")
