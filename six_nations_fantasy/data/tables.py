import pandas as pd
import datetime as dt
import itertools
import re


# TODO: extract further data from the match (matchAttacking, matchDefending)
# 
# TODO: create dataframe for match data
def extract_player_data(player, substitutions, match_id, substitute):

    # if we have found substitution data for this player, replace with 
    # that, otherwise default of 0 for substitutes and 80 for starters
    # TODO could have more than 80 minutes if go into red; do we care?
    minutes_played = substitutions.get(player['name'], 0 if substitute else 80)

    return {
                'match_id': match_id, 
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
                'offloads': int(player['offload']['value']),
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

# read match events to extract minutes played data
def parse_match_events(events):
    minutes_played = {}

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

    return minutes_played

def match_json_to_DataFrame(json_data, match_id=None):

    
    # TODO probably best to create a separate function to process this
    # ESPN-specific JSON structure
    game = json_data['gamePackage']

    player_data = game['matchLineUp']

    # hacky way to get the year: it will be the last four digits of the game header
    year = game['gameStrip']['header'][-4:]
    day, month = game['gameStrip']['date'].split('/')

    date = dt.date(int(year), int(month), int(day))


    events = game['matchCommentary']['events']
    minutes_played = parse_match_events(events)

    stats_table = game['matchStats']['table']
    attacking_table = game['matchAttacking']['col'][1][0]['data']

    # scrums and lineouts data
    defending_table = game['matchDefending']['col'][0]

    # penalties conceded information
    discipline_table = game['matchDiscipline']['col'][0][0]['data']

    match_data = {
            'match_id': match_id,
            'date': date,
            'home_team': game['matchStats']['home']['name'],
            'away_team': game['matchStats']['away']['name'],
            'home_score': game['gameStrip']['teams']['home']['score'],
            'away_score': game['gameStrip']['teams']['away']['score'],
            'home_possesion': game['matchStats']['dataVis'][0]['homeValue'],
            'away_possesion': game['matchStats']['dataVis'][0]['awayValue'],
            'home_territory': game['matchStats']['dataVis'][1]['homeValue'],
            'away_territory': game['matchStats']['dataVis'][1]['awayValue'],
            'home_runs': stats_table[0]['homeValue'],
            'away_runs': stats_table[0]['awayValue'],
            'home_meters': stats_table[1]['homeValue'],
            'away_meters': stats_table[1]['awayValue'],
            'home_clean_breaks': stats_table[2]['homeValue'],
            'away_clean_breaks': stats_table[2]['awayValue'],
            'home_won_tackles': stats_table[3]['homeValue'],
            'away_won_tackles': stats_table[3]['awayValue'],
            'home_missed_tackles': stats_table[4]['homeValue'],
            'away_missed_tackles': stats_table[4]['awayValue'],
            'home_knock_ons': stats_table[5]['homeValue'],
            'away_knock_ons': stats_table[5]['awayValue'],
            'home_defenders_beaten': attacking_table[3]['homeValue'],
            'away_defenders_beaten': attacking_table[3]['awayValue'],
            'home_penalties_conceded': discipline_table['homeTotal'],
            'away_penalties_conceded': discipline_table['awayTotal'],
            # TODO choose which stats we care about, there are couple more stuff available
            }

    starters = [
            extract_player_data(p, minutes_played, match_id, substitute=False)
            for p in itertools.chain(
                player_data['home']['team'],
                player_data['away']['team'],
                )]

    substitutes = [
            extract_player_data(p, minutes_played, match_id, substitute=True)
            for p in itertools.chain(
                player_data['home']['reserves'],
                player_data['away']['reserves'],
                )]

    return match_data, pd.DataFrame(itertools.chain(starters, substitutes))
