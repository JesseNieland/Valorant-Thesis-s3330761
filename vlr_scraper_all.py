# VLR Match Scraper
# Scrapes all matches from Valorant Champions Tour 2024
# Stores using Pandas

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# URL of the VCT 2024 main events page
vct_years = ["2023", "2024"]
base_url = "https://www.vlr.gg"
exclusions = ["team-international", "showmatch", "tarik"]

def get_event_links():
    """Scrape all event URLs from the VCT 2023/2024 overview page."""
    event_links = []

    for year in vct_years:
        vct_events_url = f"{base_url}/vct-{year}"  # Dynamic URL for each year
        response = requests.get(vct_events_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all event cards based on the class name
        events = soup.find_all("a", class_="wf-card mod-flex event-item")

        for event in events:
            # Extract the href attribute and build the full URL
            event_url = event["href"]
            event_id = event_url.split('/')[2]  # Extract event ID
            matches_url = f"{base_url}/event/matches/{event_id}/?series_id=all"
            event_links.append(matches_url)

            print(f"Found event for {year}: {matches_url}")

    return event_links

def get_match_links(event_url):
    """Fetch all individual match URLs from a given event's matches page."""
    response = requests.get(event_url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    match_links = []

    # Find all match items; match-item class should cover both variants
    matches = soup.find_all("a", class_=re.compile(r"^wf-module-item match-item.*"))

    for match in matches:
        match_url = base_url + match["href"]  # Construct full match URL

        # Skip showmatches
        if any(exclusion in match_url.lower() for exclusion in exclusions):
            print(f"Skipping showmatch: {match_url}")
            continue

        match_links.append(match_url)

    return match_links

def get_match_details(match_url):
    """
    Scrapes match details from individual match page.
    """
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
    map_notes = soup.find_all("div", class_="match-header-note")
    maps_pick_ban = None

    for note in map_notes:
        # Check if the note contains "ban" or "pick" as indicators of map veto
        if "ban" in note.text.lower() or "pick" in note.text.lower():
            maps_pick_ban = note.text.strip()
            break

    if not maps_pick_ban:
        maps_pick_ban = "No map veto information available"

    # Extract Scores per Map
    maps = []
    durations = []
    sides_chosen_by_non_picker = []  # List to store attack/defense choice of the non-picking team
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

            # Determine the side picked by the non-picker
            team1_non_picker = game.find("span", class_=re.compile("mod-2"))
            team2_non_picker = game.find("span", class_=re.compile("mod-1"))

            non_picker_sides = game.find_all("span", class_=re.compile(r"mod-(t|ct)"))  # Find all sides

            if team1_non_picker:  # Team 1 picked the map, so Team 2 chooses the side
                if non_picker_sides:  # Determine whether the first role is attack or defense
                    if "mod-t" in non_picker_sides[0]["class"]:  # First role is attack
                        sides_chosen_by_non_picker.append("Team 1: Attack")
                    elif "mod-ct" in non_picker_sides[0]["class"]:  # First role is defense
                        sides_chosen_by_non_picker.append("Team 1: Defense")
            elif team2_non_picker:  # Team 2 picked the map, so Team 1 chooses the side
                if non_picker_sides:  # Determine whether the first role is attack or defense
                    if "mod-t" in non_picker_sides[2]["class"]:  # First role is attack
                        sides_chosen_by_non_picker.append("Team 2: Attack")
                    elif "mod-ct" in non_picker_sides[2]["class"]:  # First role is defense
                        sides_chosen_by_non_picker.append("Team 2: Defense")
            else:
                if non_picker_sides:  # Determine whether the first role is attack or defense
                    if "mod-t" in non_picker_sides[0]["class"] and "mod-ct" in non_picker_sides[2]["class"]:  # First role is attack
                        sides_chosen_by_non_picker.append("Team 1: Attack & Team 2: Defense")
                    elif "mod-ct" in non_picker_sides[0]["class"] and "mod-t" in non_picker_sides[2]["class"]:  # First role is defense
                        sides_chosen_by_non_picker.append("Team 1: Defense & Team 2: Attack")

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
        "Sides Chosen by Non-Picker": "; ".join(sides_chosen_by_non_picker),
        "Match URL": match_url
    }

def scrape_all_vct_matches():
    """Scrape all matches from all events in VCT 2023/2024 and save to CSV."""
    all_matches = []

    # Get all event URLs
    event_links = get_event_links()
    print(f"Found {len(event_links)} events.")

    # Loop through events and fetch match URLs
    for event_url in event_links:
        print(f"Scraping matches from: {event_url}")
        match_links = get_match_links(event_url)

        time.sleep(1)  # Respect rate limits

        # Loop through matches and scrape details
        for match_url in match_links:
            print(f"Scraping match: {match_url}")
            match_data = get_match_details(match_url)
            all_matches.append(match_data)

            # Respect rate limits by sleeping
            time.sleep(1)

    # Save all match data to CSV, separated by year
    for year in vct_years:
        year_matches = [match for match in all_matches if year in match["Match URL"]]
        df = pd.DataFrame(year_matches)
        df.to_csv(f"vct_{year}_matches_2.csv", index=False)
        print(f"All data for {year} saved to vct_{year}_matches.csv")

def scrape_vct_test():
    """Scrape a small number of events and matches for testing."""
    all_matches = []

    event_links = get_event_links()[:2]  # Limit to the first 2 events
    for event_url in event_links:
        print(f"Scraping matches from: {event_url}")
        match_links = get_match_links(event_url)[:2]  # Limit to 2 matches per event

        time.sleep(1)  # Respect rate limits

        for match_url in match_links:
            print(f"Scraping match: {match_url}")
            match_data = get_match_details(match_url)
            all_matches.append(match_data)
            time.sleep(1)  # Respect rate limits

    df = pd.DataFrame(all_matches)
    print(df.head())

# Run the scraper
scrape_all_vct_matches()

# df_2023 = pd.read_csv("vct_2023_matches_2.csv")
# df_2024 = pd.read_csv("vct_2024_matches_2.csv")

# combined_df = pd.concat([df_2023, df_2024], ignore_index=True)
# combined_df.to_csv("vct_2023_2024_matches_2.csv", index=False)

# Get all event links
# event_links = get_event_links()
# print(f"Found {len(event_links)} events.")
# print(event_links)

# Get all match links from a specific event
# match_links = get_match_links("https://www.vlr.gg/event/matches/1924/?series_id=all")
# print(f"Found {len(match_links)} matches.")
# print(match_links[:5])

# # Get all match data from a specific match
# match_data = get_match_details("https://www.vlr.gg/219596/number-one-player-vs-douyu-gaming-valorant-champions-2023-china-qualifier-r1")
# print(match_data)

# Tests the Scraper and pulls only the first few events and matches
# scrape_vct_test()
