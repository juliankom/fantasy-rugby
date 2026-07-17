import sys
import csv
import pandas as pd
from pulp import *

def add_position_constraint(problem, df): 

    # number of players allowed to be selected in each position
    allowed_players = {
            'Hooker': 1,
            'Prop': 2,
            'Second-row': 2,
            'Back-row': 3,
            'Scrum-half': 1,
            'Fly-half': 1,
            'Centre': 2,
            'Back Three': 3,
            }

    # constrain the position selections to not exceed their maxima
    for pos, group in df.groupby('nominal_pos'):
        if pos != 'Bench':
            problem += lpSum(group['selected_var']) <= allowed_players[pos]

    # 15 field players may be chosen
    problem += lpSum(df['selected_var']) <= 15
    
def add_team_constraint(problem, df, max_per_team=4):
    for pos, group in df.groupby('team'):
        problem += lpSum(group['selected_var']) <= max_per_team

def add_captaincy_constraint(problem, df):
    # at most one captain
    problem += lpSum(df['captain_var']) <= 1

    # the captain must be selected
    for x, c in df[['selected_var', 'captain_var']].values:
        problem += c <= x

def add_supersub_constraint(problem, df):
    problem += lpSum(df['supersub_var']) <= 1

    # the supersub must not be selected
    for x, s in df[['selected_var', 'supersub_var']].values:
        problem += s <= 1 - x

def add_budget_constraint(problem, df, budget=200):
    problem += lpSum(df['cost'] * df['selected_var']) <= budget

def create_variables(df):
    # create a binary variable x_i for every player, meaning "selected"
    df['selected_var'] = df.apply(lambda p: LpVariable(f"x{p.name}", cat='Binary'), axis=1)
    
    # create a binary variable c_i for every player, meaning "captain"
    df['captain_var'] = df.apply(lambda p: LpVariable(f"c{p.name}", cat='Binary'), axis=1)

    # create a binary variable s_i for every player, meaning "supersub"
    df['supersub_var'] = df.apply(lambda p: LpVariable(f"s{p.name}", cat='Binary'), axis=1)
    
# given a dataframe with rows being players with cost, score (this could be the actual
# matchday score or a prediction), nominal_pos and team, find the 
# optimal lineup for fantasy 
def setup_problem(df):

    create_variables(df)
    problem = LpProblem('optimal_fantasy_selection', LpMaximize)

    add_position_constraint(problem, df)
    add_team_constraint(problem, df)
    add_captaincy_constraint(problem, df)
    add_supersub_constraint(problem, df)
    add_budget_constraint(problem, df)

    # add the score of selected players, with a multiplier of 2 if additionally captain, and
    # add the score of the supersub, with multiplier of 3 if a substitute, 1/2 if not
    problem += lpSum(df['score'] * (df['selected_var'] + df['captain_var']) \
        + df['score'] * df['supersub_var'] * (3 * df['is_sub'] + 1/2 * (1 - df['is_sub'])))

    return problem
