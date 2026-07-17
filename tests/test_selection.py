import csv
import re
import pandas as pd
import unicodedata
from selection import *

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

def read_csv(filename): 
    df = pd.read_csv(filename, \
        usecols=range(1, 6), \
        names=['cost', 'score', 'nominal_pos', 'team', 'name'])

    df['is_sub'] = df['nominal_pos'] == 'Bench'
    df = df.apply(process_name, axis=1)

    return df

if __name__ == '__main__':
    # first argument is expected to be the name of a csv file containing round data
    filename = sys.argv[1]

    # TODO: calculate scores manually using our statistics
    df = read_csv(filename)
    
    problem = setup_problem(df)
    solver = COIN_CMD()

    status = problem.solve(solver)

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

    print(team)
    print(f'total score: {team["final_score"].sum()}')
    print(f'total cost: {team["cost"].sum()}')


