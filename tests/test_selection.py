import csv
import re
import pandas as pd
import unicodedata
import datetime
from six_nations_fantasy.ilp import *
from six_nations_fantasy.score import score
from six_nations_fantasy.models.weighted_mean import *

# turn e.g. Mickaël into Mickael 
def remove_accents(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def process_name(row):
        name = remove_accents(row['name'])

        # match anything (not containing brackets) inside (...) 
        match = re.search(r'\(([^()]*)\)\s*$', name)

        position_abbrevs = {
                'H': 'Hooker', 
                'P': 'Prop',
                'SR': 'Second-row',
                'BR': 'Back-row',
                'SH': 'Scrum-half',
                'FH': 'Fly-half',
                'C': 'Centre',
                'BT': 'Back Three'
                }

        if match:
            # keep only the part of the name up to the (...) and strip whitespace
            name = name[:match.start()].strip()

            label = match.group(1)

            if label in position_abbrevs and row['is_sub']:
                row['nominal_pos'] = position_abbrevs[label]

        # update the name after stripping of accents and labels
        row['name'] = name

        return row

def read_matchday_csv(filename): 
    df = pd.read_csv(filename, \
        usecols=range(1, 6), \
        names=['cost', 'score', 'nominal_pos', 'team', 'name'])

    df['is_sub'] = df['nominal_pos'] == 'Bench'
    df = df.apply(process_name, axis=1)

    return df

def find_selected_team(df):
    team = df[(df['selected_var'].apply(value) == 1) 
              | (df['supersub_var'].apply(value) == 1)]

    team['is_captain'] = team.apply(lambda x: value(x['captain_var']), axis=1) 
    team['is_supersub'] = team.apply(lambda x: value(x['supersub_var']), axis=1) 

    team = team[['name', 'cost', 'score', 'nominal_pos', 'team', \
            'is_sub', 'is_captain', 'is_supersub']]

    team['final_score'] = team['score'] * (1 + team['is_captain']) \
            * (1 - team['is_supersub']) \
            + team['score'] * team['is_supersub'] * \
            (3 * team['is_sub'] + 1/2 * (1 - team['is_sub']))

    return team

def run_solver(matchday_data):
    problem = setup_problem(matchday_data)
    solver = COIN_CMD(msg=False)

    status = problem.solve(solver)
    team = find_selected_team(matchday_data)

    # TODO: annoying problem: the most important data we are missing are
    # probably breakdown steals (given for penalties also), 50-22s and kicks retained
    print(team)
    print(f'total score: {team["final_score"].sum()}')
    print(f'total cost: {team["cost"].sum()}')

if __name__ == '__main__':
    # first argument is expected to be the name of a csv file containing round data
    filename = sys.argv[1]

    # TODO: calculate scores manually using our statistics
    matchday_data = read_matchday_csv(filename)

    matches = pd.read_parquet('all_data_matches.parquet')
    perfs = pd.read_parquet('all_data_performances.parquet')

    # TODO: we are just assuming that we are testing with the first matchday of 2026
    d0 = datetime.date(2026, 2, 5)
    d1 = datetime.date(2026, 2, 7)
    fm = matches[(d0 <= matches['date']) & (matches['date'] <= d1)]

    # for the players in the matchday data, calculate the score based on the 
    # performance states imported
    md = pd.merge(matchday_data, perfs[perfs['match_id'].isin(fm.index)], on='name')
    matchday_data['score'] = md.apply(score, axis=1)

    #print(
    #    matchday_data
    #    .assign(diff=abs(matchday_data['c_score'] - matchday_data['score']))
    #    .nlargest(20, 'diff')
    #)

    # OBSERVATION: a calculated score of 604 (574 after ignoring forward try bonus) 
    # was optimal for the first weekend;
    # this same team would have acheived a score of 714 using the actual
    # weekend scores, showing by how much our scoring is off.
    # 
    # 45 of these points is explained by player-of-the-match bonuses, the 
    # remaining are probably breakdown steals, 50-22s and kicks retained

    run_solver(matchday_data)

    merged = perfs.join(matches, on='match_id')
    training_data = merged[merged['date'].apply(lambda d: d.year) < 2026]

    # now use our naive model to calculate the scores
    model = WeightedMean(score, training_data, datetime.date(2025, 3, 15), decay_factor=0.99)
    matchday_data['score'] = md.apply(lambda p: model.estimate(p['name']), axis=1)

    run_solver(matchday_data)

    # TODO make some notebooks
    # RESULT: weighted mean acheives 444.2 / 574 (77.4%)
    # scaling up yield 524.9 / 714 or 554.3 / 754
