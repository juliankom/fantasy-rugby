import json
import os
from six_nations_fantasy.data.tables import *

def dataframe_from_filename(filename, directory):
    with open(directory + '/' + filename) as f:
        json_data = json.load(f)

    return match_json_to_DataFrame(json_data, match_id=filename.removesuffix('.json'))

def database_from_dir(directory):
    dirs = os.listdir(directory)

    # TODO we are assuming that all files in the directory are JSONs, better to check
    return pd.concat([dataframe_from_filename(f, directory) for f in dirs], 
                     ignore_index=True)

