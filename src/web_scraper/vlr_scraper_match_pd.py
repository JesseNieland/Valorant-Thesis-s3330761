# VLR Match Scraper
# Scrapes one match from Valorant Champions Tour 2024
# This only uses the match site
# Stores using Pandas

import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the match page to scrape
match_url1 = "https://www.vlr.gg/378829/edward-gaming-vs-team-heretics-valorant-champions-2024-gf"

def get_match_details(match_url):
    # Fetch the page content
    response = requests.get(match_url, timeout=10)
    response.raise_for_status()  # Raise an exception for HTTP errors
    soup = BeautifulSoup(response.content, "html.parser")

    event_element = soup.find("a", class_="match-header-event")
    event_name = event_element.find("div", style="font-weight: 700;").text.strip()

    # Extract Team Names
    teams = [team.text.strip() for team in soup.find_all("div", class_="wf-title-med")]

    # Extract Match Date and Time
    date_time = soup.find("div", class_="moment-tz-convert")["data-utc-ts"]

    # Extract Match Format (e.g., Bo3, Bo5)
    notes = soup.find_all("div", class_="match-header-vs-note")
    match_format = notes[1].text.strip() if len(notes) > 1 else "Unknown"

    # Extract Maps Picked and Banned
    maps_pick_ban = soup.find("div", class_="match-header-note").text.strip()

    # Extract Scores per Map
    maps = []
    durations = []
    for game in soup.find_all("div", class_="vm-stats-game"):
        try:
            map_div = game.find("div", class_="map")
            if map_div is None:
                continue

            map_name_raw = ' '.join(map_div.text.strip().split()).replace(" PICK", "")

            # Extract match duration (if available)
            map_name, duration = map_name_raw.rsplit(" ", 1)
            durations.append(duration)

            scores = game.find_all("div", class_="score")
            if len(scores) < 2:
                continue

            team1_score = scores[0].text.strip()
            team2_score = scores[1].text.strip()
            maps.append(f"{map_name}: {team1_score}-{team2_score}")
        except AttributeError as e:
            print(f"Error extracting map data: {e}")

    # Extract Scores per Half (Attack/Defense) for each map
    halves = []
    for game in soup.find_all("div", class_="vm-stats-game"):
        if game.get("data-game-id") == "all":
            continue

        try:
            team1_attack = game.find("span", class_="mod-t").text.strip()
            team1_defense = game.find("span", class_="mod-ct").text.strip()
            team2_defense = game.find_all("span", class_="mod-ct")[1].text.strip()
            team2_attack = game.find_all("span", class_="mod-t")[1].text.strip()

            map_div = game.find("span", class_="mod-ot")
            if map_div is None:
                team1_half = f"{team1_attack} / {team1_defense}"
                team2_half = f"{team2_defense} / {team2_attack}"
            else:
                team1_overtime = map_div.text.strip()
                team2_overtime = game.find_all("span", class_="mod-ot")[1].text.strip()
                team1_half = f"{team1_attack} / {team1_defense} / {team1_overtime}"
                team2_half = f"{team2_defense} / {team2_attack} / {team2_overtime}"

            halves.append(f"Team 1: {team1_half}, Team 2: {team2_half}")

        except (IndexError, AttributeError) as e:
            print(f"Error extracting half scores: {e}")

    # Prepare the data for pandas
    return {
        "Event": event_name,
        "Team 1": teams[0] if len(teams) > 0 else None,
        "Team 2": teams[1] if len(teams) > 1 else None,
        "Date": date_time,
        "Match Format": match_format,
        "Maps Picked/Banned": maps_pick_ban,
        "Score per Map": "; ".join(maps),
        "Match Durations": "; ".join(durations),
        "Score per Half": "; ".join(halves),
        "Match URL": match_url
    }

# Scrape the match and store the data in a DataFrame
match_data = get_match_details(match_url1)
df = pd.DataFrame([match_data])

df.to_csv("vct_match.csv", index=False)
print("Data saved to vct_match.csv")
