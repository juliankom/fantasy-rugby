import pandas as pd
import os

def select_matches(matches):
    accepted_competitions = set([
        'Six Nations', 
        'International Test Match', 
        'Internationals',
        'United Rugby Championship',
        'French Top 14',
        'Top 14',
        'Gallagher Prem',
        'Premiership',
    ])

    return matches[matches.apply(lambda m: m['competition'] in accepted_competitions)]