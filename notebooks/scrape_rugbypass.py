import requests
import pandas as pd
import datetime
from bs4 import BeautifulSoup

stats = [
    "carries",
    "clean_breaks", # displayed as line breaks
    "completed_tackles",
    "turnovers_conceded",
    "turnovers_won",
    # attacking stats
    "total_kicks",
    "passes",
    "carries_metres",
    "offloads",
    "defenders_beaten",
    "try_assists",
    "tries",
    "carries_per_minute",
    # defending stats
    "total_tackles", # displayed as tackles made
    "missed_tackles",
    "dominant_tackles",
    "ruck_turnovers",
    "lineouts_won",
    "tackles_per_minute",
    # discipline stats
    "red_cards",
    "yellow_cards",
    "penalties_conceded"
]

def process_selected_stat(response, dtype=int):
    data = response.json()
    soup = BeautifulSoup(data['html'], "html.parser")

    players = soup.select("div.player")

    stats = {}
    for player in players:
        name = player.select_one(".name").text.strip()
        value = dtype(player.select_one(".total").text.strip())

        stats[name] = value

    return stats
    
def scrape_stats(session, match_url):
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.rugbypass.com",
        "Referer": "https://www.rugbypass.com/",
    })
    
    payload = {
        "action": "filter-players-stats",
        "team": "0",
    #    "stat": "carries",
        "isContent": "1",
    }
    
    data = {}
    
    for stat in stats:
        payload["stat"] = stat
        response = session.post(match_url, files={k: (None, v) for k, v in payload.items()})
        if stat in set(["tackles_per_minute", "carries_per_minute"]):
            data[stat] = process_selected_stat(response, dtype=float)
        else:
            data[stat] = process_selected_stat(response)
            
    return data

def extract_minute(text):
    try:
        x = int(text[:2])
        if x > 80:
            x = 80
    except ValueError:
        x = int(text[:1])
    return x

def calculate_minutes_played(changes, starter=True):
    # whether the player finished the game in the same position as they started
    # (i.e. did a starter finish on the pitch or a sub finish on the bench)
    same = len(changes) % 2 == 0
    
    if (starter and same) or (not starter and not same):
        changes.append(80)
        
    with_signs = [x * (2*(i % 2 != starter) - 1) for i, x in enumerate(changes)]
    return sum(with_signs)
            
def process_team(team):
    players = team.select("div.player")

    data = {}
    
    for player in players:
        name = player.select_one(".name").get_text(strip=True)
        number = int(player.select_one(".num").text.strip())

        subs = [
            extract_minute(sub.get_text(strip=True))
            for sub in player.select_one(".sub").find_all("div")
        ]

        mins = calculate_minutes_played(subs, starter=(number <= 15))

        data[name] = {"number": number, "minutes_played": mins}
        
    return data

def process_teams(response):
    soup = BeautifulSoup(response.text, "html.parser")
    teams = soup.select("div.players")

    return {
        "home": process_team(teams[0]),
        "away": process_team(teams[1])
    } 

def scrape_teams(session, teams_url):
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.rugbypass.com",
        "Referer": "https://www.rugbypass.com/",
    })

    response = session.get(teams_url)

    return process_teams(response)

month_dict = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

def process_base(response, competition_id):
    soup = BeautifulSoup(response.text, "html.parser")

    lines = [l.select_one(".title").get_text(strip=True) 
        for l in soup.select_one(".match-details").select(".line")
    ]

    teams = lines[0]
    home, away = teams.split(" v ")
    home = home.strip()
    away = away.strip()
    
    extra_info = lines[1]
    _, number, month, year, _, _ = lines[2].split()

    month = month_dict[month]
    year = int(year[:4])
    try:
        day = int(number[:2])
    except ValueError:
        day = int(number[:1])

    return {
        "home_team": home,
        "away_team": away,
        "date": datetime.date(year, month, day),
        "competition_id": competition_id,
        "extra_info": extra_info,
    }

def scrape_base(session, base_url, competition_id):
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.rugbypass.com",
        "Referer": "https://www.rugbypass.com/",
    })

    response = session.get(base_url)

    return process_base(response, competition_id)

def string_as_int_or_percent(text):
    if "%" in text:
        return float(text.replace("%", "")) / 100
    elif ":" in text:
        a, b = text.split(":")
        return float(a) / (float(a) + float(b))
    return int(text)

def format_label(text):
    return text.strip().lower().replace(" ", "_")
    
def process_match_stats(response):
    soup = BeautifulSoup(response.text, "html.parser")

    data = {} 
    for s in soup.select("div.stat"):
        name = s.select_one(".mid").get_text(strip=True)
    
        if name != "":
            hv, av = s.get_text(strip=True).split(name)
            hv = hv.replace("m", "")
            av = av.replace("m", "")
            
            data[format_label(name)] = {
                "home": string_as_int_or_percent(hv),
                "away": string_as_int_or_percent(av)
            }

    for s in soup.select("div.percents"):
        x0, x1, x2, x3 = [stat.get_text(strip=True) for stat in s.find_all("div")]
        data[s["id"].replace("canvas_", "")] = {
            "0": string_as_int_or_percent(x0),
            "1": string_as_int_or_percent(x1),
            "2": string_as_int_or_percent(x2),
            "3": string_as_int_or_percent(x3)
        }
        
    return data