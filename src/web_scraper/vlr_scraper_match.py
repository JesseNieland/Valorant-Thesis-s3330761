# VLR Match Scraper
# Scrapes one match from Valorant Champions Tour 2024
# This only uses the match site

import requests
from bs4 import BeautifulSoup

# URL of the match page to scrape
match_url1 = "https://www.vlr.gg/378829/edward-gaming-vs-team-heretics-valorant-champions-2024-gf"

def get_match_details(match_url):
    # Fetch the page content
    response = requests.get(match_url, timeout=10)
    response.raise_for_status()  # Raise an exception for HTTP errors
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract Team Names
    teams = [team.text.strip() for team in soup.find_all("div", class_="wf-title-med")]
    print("Teams:", teams)

    # Extract Match Date and Time
    date_time = soup.find("div", class_="moment-tz-convert")["data-utc-ts"]
    print("Date:", date_time)

    # Extract Match Format (e.g., Bo3, Bo5)
    notes = soup.find_all("div", class_="match-header-vs-note")
    if len(notes) > 1:
        match_format = notes[1].text.strip()  # Get the second occurrence
    else:
        match_format = "Unknown"  # Fallback if the second element doesn't exist
    print("Match Format:", match_format)

    # Extract Maps Picked and Banned
    maps_pick_ban = soup.find("div", class_="match-header-note").text.strip()
    print("Maps Picked/Banned:", maps_pick_ban)



    # Extract Scores per Map Correctly
    maps = []
    for game in soup.find_all("div", class_="vm-stats-game"):
        try:
            # Attempt to find the map name
            map_div = game.find("div", class_="map")
            if map_div is None:
                continue

            map_name_raw = map_div.text.strip()
            # Clean up the map name
            map_name = ' '.join(map_name_raw.split())

            scores = game.find_all("div", class_="score")
            if len(scores) < 2:
                continue  # Skip if scores are missing (e.g., unplayed maps)

            team1_score = scores[0].text.strip()
            team2_score = scores[1].text.strip()
            maps.append(f"{map_name}: {team1_score}-{team2_score}")
        except AttributeError as e:
            print(f"Error extracting map data: {e}")
    print("Score per Map:", maps)

    # Extract Scores per Half (Attack/Defense) for each map
    # NOT the first side is first, attack is first for team 1
    halves = []
    for game in soup.find_all("div", class_="vm-stats-game"):
        # Check if the data-game-id is "all"
        if game.get("data-game-id") == "all":
            continue  # Skip this game if data-game-id is "all"
        try:
            # Find exactly two half scores per map (attack/defense)
            team1_attack = game.find("span", class_="mod-t").text.strip()
            team1_defense = game.find("span", class_="mod-ct").text.strip()
            team2_defense = game.find_all("span", class_="mod-ct")[1].text.strip()
            team2_attack = game.find_all("span", class_="mod-t")[1].text.strip()

            map_div = game.find("span", class_="mod-ot")
            if map_div is None:
                # Combine the scores in the correct order
                team1_half = f"{team1_attack} / {team1_defense}"
                team2_half = f"{team2_defense} / {team2_attack}"

                halves.append(f"Team 1: {team1_half}, Team 2: {team2_half}")
            else:
                team1_overtime = game.find("span", class_="mod-ot").text.strip()
                team2_overtime = game.find_all("span", class_="mod-ot")[1].text.strip()

                # Combine the scores in the correct order
                team1_half = f"{team1_attack} / {team1_defense} / {team1_overtime}"
                team2_half = f"{team2_defense} / {team2_attack} / {team2_overtime}"

                halves.append(f"Team 1: {team1_half}, Team 2: {team2_half}")


        except (IndexError, AttributeError) as e:
            print(f"Error extracting half scores: {e}")

    print("Score per Half:", halves)

# Run the function to test the scraping
get_match_details(match_url1)
