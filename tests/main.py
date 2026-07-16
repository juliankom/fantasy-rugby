import json
import os
import pandas as pd
import datetime as dt
from six_nations_fantasy.models.naive_mean import *
from six_nations_fantasy.models.weighted_mean import *
from tables import *

# (FB) Fullbacks, (W)ings, (C)enters, (FH) Fly halves, (SH) Scrum halves
backs_positions = set(['FB', 'W', 'C', 'FH', 'SH'])

# (P)rops, (H)ookers, (L)ocks, (FL)ankers, (N8) Number 8s
forwards_positions = set(['P', 'H', 'L', 'FL', 'N8'])

# TODO should try to move away from per-row operations and focus on 
# working with the columns directly (maybe)?
def score_common(player): 
    return 4 * player['assists'] \
        + 2 * player['conversions'] \
        + 3 * player['penalties_kicked'] \
        + 5 * player['drop_goals'] \
        + 2 * player['defenders_beaten'] \
        + player['meters_carried'] // 10 \
        + player['tackles'] \
        + 7 * player['lineout_steals'] \
        - 5 * player['yellow_cards'] \
        - 8 * player['red_cards']

def score_back(player):
    return 10 * player['tries'] + score_common(player)

def score_forward(player):
    return 15 * player['tries'] + score_common(player)

def score(player):
    if player['position'] in backs_positions:
        return score_back(player)
    elif player['position'] in forwards_positions:
        return score_forward(player)
    else:
        # TODO what to do about substitutes?
        # Well, this again depends on the database of _nominal_ positions,
        # which we don't have at this point...
        return 0

def dataframe_from_filename(filename, directory):
    with open(directory + '/' + filename) as f:
        json_data = json.load(f)
    df = match_json_to_DataFrame(json_data, match_id=filename.removesuffix('.json'))
    return df

def database_from_dir(directory='training_data'):
    dirs = os.listdir(directory)
    return pd.concat([dataframe_from_filename(f, directory) for f in dirs], 
                     ignore_index=True)

training_data = database_from_dir()

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
