"""Clean raw ESPN html into more managable JSON files
"""
import json
import re
import sys
import requests
from bs4 import BeautifulSoup

def clean(html):
    soup = BeautifulSoup(html, 'html.parser')

    # find the location of the JSON containing all data we want
    script = soup.find('script', text = re.compile('window.__INITIAL_STATE__'))

    # regex to isolate the raw JSON
    json_text = re.search(r'^\s*window.__INITIAL_STATE__\s*=\s*({.*?})\s*;\s*$',\
        script.string, flags=re.DOTALL | re.MULTILINE).group(1)

    data = json.loads(json_text)

    return data

def get_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Connection": "keep-alive",
    }

    session = requests.Session()
    session.headers.update(headers)

    session.get("https://espn.com")
    response = session.get(url, timeout=10)

    return response.text

def get_name(data, year):
    home_team_name = data['gamePackage']['gameStrip']['teams']['home']['abbrev']
    away_team_name = data['gamePackage']['gameStrip']['teams']['away']['abbrev']

    return str(year) + "_" + home_team_name + "_" + away_team_name 

def write_json(data, year):
    game_name = get_name(data, year)

    with open(f'games/{game_name}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def html_to_json(filename):
    with open(filename) as html:
        data = clean(html)

    game_name, _ = filename.split('.')

    with open(f'{game_name}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    year = int(sys.argv[1])

    for filename in sys.argv[2:]:
        with open(filename) as html:
            data = clean(html)

        game_name = get_name(data, year)

        with open(f'newgames/{game_name}.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
