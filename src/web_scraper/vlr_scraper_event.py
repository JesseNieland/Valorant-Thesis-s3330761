# VLR Match Scraper
# Scrapes all matches from Valorant Champions Tour 2024 Pacific Kickoff tourney
# Should scrape every individual match url from the tourney

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

# Base URL for the event page
event_url = "https://www.vlr.gg/event/matches/1924/champions-tour-2024-pacific-kickoff/?series_id=all"
base_url = "https://www.vlr.gg"

# Rate limit to 1 request per second to stay on the safe side
RATE_LIMIT_SECONDS = 1

def get_match_links(event_url):
    """
    Gets all match links from the event page.
    """
    response = requests.get(event_url, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all match items on the event page
    matches = soup.find_all("a", class_="wf-module-item")
    match_links = [base_url + match["href"] for match in matches]
    return match_links

def get_match_details(match_url):
    """
    Scrapes match details from individual match page.
    """
    response = requests.get(match_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract teams
    team_names = [team.text.strip() for team in soup.find_all("div", class_="match-header-vs-team-name")]

    # Extract date and time of the match
    date_time = soup.find("div", class_="match-header-date").text.strip()

    # Extract score
    scores = [score.text.strip() for score in soup.find_all("div", class_="match-header-vs-score")]

    # Extract round details for each map (attack/defense performance)
    rounds = []
    for map_div in soup.find_all("div", class_="vm-stats-game"):
        map_name = map_div.find("div", class_="map").text.strip()
        team1_score = map_div.find("div", class_="score").find_all("span")[0].text.strip()
        team2_score = map_div.find("div", class_="score").find_all("span")[1].text.strip()
        rounds.append(f"{map_name}: {team1_score}-{team2_score}")

    # Create a dictionary with match details
    match_data = {
        "Team 1": team_names[0],
        "Team 2": team_names[1],
        "Date and Time": date_time,
        "Team 1 Final Score": scores[0],
        "Team 2 Final Score": scores[1],
        "Round Details": "; ".join(rounds),
        "Match URL": match_url
    }
    return match_data

def scrape_all_matches(event_url):
    """
    Gathers the data from all matches.
    """
    match_links = get_match_links(event_url)
    all_matches = []

    for link in match_links:
        print(f"Scraping match: {link}")
        match_data = get_match_details(link)
        all_matches.append(match_data)
        time.sleep(RATE_LIMIT_SECONDS)  # Respect rate limits

    return all_matches

def save_to_csv(data, filename="matches.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Run the scraper
if __name__ == "__main__":
    matches = scrape_all_matches(event_url)
    save_to_csv(matches)
    print("Scraping completed.")