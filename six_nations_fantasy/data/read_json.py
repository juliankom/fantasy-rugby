import json
import os
from six_nations_fantasy.data.tables import *

def dataframe_from_filename(filename, directory, match_id=None):
    if match_id is None:
        match_id = filename.removesuffix('.json')

    with open(directory + '/' + filename) as f:
        json_data = json.load(f)

    return match_json_to_DataFrame(json_data, match_id)

def database_from_dir(directory):
    files = os.listdir(directory)

    extracted_data = [dataframe_from_filename(f, directory, i) for i, f \
        in enumerate(files) if f.endswith('.json')]
    
    matches = pd.DataFrame([m for m, ps in extracted_data])
    players = pd.concat([ps for m, ps in extracted_data])

    return matches, players

