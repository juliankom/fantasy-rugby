import re
import sys
import unicodedata
from math import log
from pathlib import Path

import pandas as pd
from pulp import COIN_CMD, value

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.ilp import setup_problem
from six_nations_fantasy.models import RecencyWeightedMean
from six_nations_fantasy.score import simple_score

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
              | (df['supersub_var'].apply(value) == 1)].copy()

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

if __name__ == "__main__":
    matchday_data = read_matchday_csv(sys.argv[1])
    data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    dataset = load_dataset(data_dir)
    dates = dataset.matches["date"]
    training = dataset.select_matches(dataset.matches.loc[dates < "2026-01-01", "match_id"])
    round_ids = dataset.matches.loc[dates.between("2026-02-05", "2026-02-07"), "match_id"]
    first_round = dataset.select_matches(round_ids).filter_competitions(["Six Nations"])
    fixtures = first_round.player_matches()

    # Match the previous experiment's 0.99 decay per week.
    model = RecencyWeightedMean(
        scorer=simple_score, half_life_days=7 * log(0.5) / log(0.99)
    )
    model.fit(training)
    scores = fixtures.assign(
        actual=simple_score(fixtures),
        predicted=model.predict(fixtures[["match_id", "player_id", "date"]]),
        team=fixtures["home_team"].where(fixtures["home"], fixtures["away_team"]),
    )
    players = pd.read_parquet(data_dir / "players.parquet")
    scores = scores.join(players["name"], on="player_id")
    # The roster and performance tables sometimes use different hyphenation.
    matchday_data["name_key"] = matchday_data["name"].str.replace("-", " ")
    scores["name_key"] = scores["name"].map(remove_accents).str.replace("-", " ")
    matchday_data = matchday_data.merge(
        scores[["team", "name_key", "actual", "predicted"]],
        on=["team", "name_key"],
        how="left",
        validate="one_to_one",
    )
    if matchday_data["actual"].isna().any():
        missing = matchday_data.loc[matchday_data["actual"].isna(), "name"]
        raise ValueError(f"Could not match players to Round 1 data: {', '.join(missing)}")

    for label, column in [("Observed scores", "actual"), ("Model predictions", "predicted")]:
        print(label)
        run_solver(matchday_data.assign(score=matchday_data[column]))
