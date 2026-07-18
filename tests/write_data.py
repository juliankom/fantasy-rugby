import argparse
import pandas as pd
from six_nations_fantasy.data.read_json import database_from_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write data to Parquet')
    parser.add_argument('directory')

    args = parser.parse_args() 

    matches, performances = database_from_dir(args.directory)

    # TODO this is a very rudimentary directory name handling 
    matches.to_parquet(f'{args.directory.strip("/")}_matches.parquet')
    performances.to_parquet(f'{args.directory.strip("/")}_performances.parquet')
