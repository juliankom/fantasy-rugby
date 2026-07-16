import pandas as pd
import datetime as dt
import itertools
import re

# TODO: reserves have position "R", but positions here are also less relvant
# since they are fixed by the fantasy game

# TODO: create dataframe for match data

def extract_player_data(player, substitutions, match_id, date, substitute):

    # if we have found substitution data for this player, replace with 
    # that, otherwise default of 0 for substitutes and 80 for starters
    # TODO could have more than 80 minutes if go into red; do we care?
    minutes_played = substitutions.get(player['name'], 0 if substitute else 80)

    # TODO we are already assuming that player names are unique for the above
    # substitutions solution, so we might as well get rid of player IDs
    return {
                'match_id': match_id, 
                'date': date,
                #'player_id': player['id'],
                'name': player['name'],
                'position': player['position'],
                'is_substitute': substitute,
                'tries': int(player['tries']['value']),
                'assists': int(player['tryassists']['value']),
                'penalties_kicked': int(player['penaltygoals']['value']),
                'conversions': int(player['conversiongoals']['value']),
                'drop_goals': int(player['dropgoalsconverted']['value']),
                'meters_carried': int(player['metres']['value']),
                'defenders_beaten': int(player['defendersbeaten']['value']),
                'lineout_steals': int(player['lineoutwonsteal']['value']),
                'turnovers_conceded': int(player['turnoversconceded']['value']),
                'tackles': int(player['tackles']['value']),
                'yellow_cards': int(player['yellowcards']['value']),
                'red_cards': int(player['redcards']['value']),
                'minutes_played': minutes_played
            }

def parse_time_format(time):
    if '+' in time:
        return 80
    return int(time.strip("'"))

def match_json_to_DataFrame(json_data, match_id=None):

    # TODO probably best to create a separate function to process this
    # ESPN-specific JSON structure
    player_data = json_data['gamePackage']['matchLineUp']

    # hacky way to get the year: it will be the last four digits of the game header
    year = json_data['gamePackage']['gameStrip']['header'][-4:]
    day, month = json_data['gamePackage']['gameStrip']['date'].split('/')

    date = dt.date(int(year), int(month), int(day))

    # read match events to extract minutes played data
    minutes_played = {}

    events = json_data['gamePackage']['matchCommentary']['events']
    for e in events:
        x = re.match(r'Player substituted - (.*) , (.*)', e['text'])

        t = parse_time_format(e['time'])

        if x is not None:
            player_name, team = x.group(1), x.group(2)
            minutes_played[player_name] = t

        x = re.match(r'Substitute on - (.*) , (.*)', e['text'])
        if x is not None:
            player_name, team = x.group(1), x.group(2)
            minutes_played[player_name] = 80 - t

    starters = [
            extract_player_data(p, minutes_played, match_id, date, substitute=False)
            for p in itertools.chain(
                player_data['home']['team'],
                player_data['away']['team'],
                )]

    substitutes = [
            extract_player_data(p, minutes_played, match_id, date, substitute=True)
            for p in itertools.chain(
                player_data['home']['reserves'],
                player_data['away']['reserves'],
                )]

    return pd.DataFrame(itertools.chain(starters, substitutes))
