# VLR Match Scraper
# Scrapes all matches from Valorant Champions Tour 2024 Pacific Kickoff tourney
# This only uses the event site and doesn't yet go into individual links for the matches

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

# URL of the event page containing all matches
url = "https://www.vlr.gg/event/matches/1924/champions-tour-2024-pacific-kickoff/?series_id=all"

# Headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
}

# Send a request to fetch the page content
response = requests.get(url, headers=headers, timeout=10)

# Check if the request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all match entries using the appropriate CSS class
    match_items = soup.find_all('a', class_='wf-module-item')

    # Prepare a list to store match data
    match_data = []

    # Extract information from each match entry
    for match in match_items:
        match_url = "https://www.vlr.gg" + match['href']
        team1 = match.find('div', class_='match-item-vs-team-name').text.strip()
        team2 = match.find_all('div', class_='match-item-vs-team-name')[1].text.strip()
        time_info = match.find('div', class_='match-item-time').text.strip()

        match_data.append({
            'Team 1': team1,
            'Team 2': team2,
            'Match Time': time_info,
            'Match URL': match_url
        })

        # Add a small delay as rate limit
        time.sleep(0.4) 

    # Convert the data to a DataFrame
    df = pd.DataFrame(match_data)
    print(df)

    df.to_csv('matches.csv', index=False)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")