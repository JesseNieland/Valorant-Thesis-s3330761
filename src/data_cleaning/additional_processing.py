import pandas as pd
import ast

# Load the datasets
match_summary = pd.read_csv('match_summary_chronological.csv')
map_data = pd.read_csv('map_data_chronological.csv')

# Convert date columns to datetime format
match_summary['Date'] = pd.to_datetime(match_summary['Date'])
map_data['Date'] = pd.to_datetime(map_data['Date'])

# Convert existing durations to integers if they exist, leave missing values as NaN
map_data['Map Duration'] = map_data['Map Duration'].apply(lambda x: int(x) if pd.notna(x) else None)

# Merge datasets on 'Match ID' to combine map-level with match-level information
map_data = map_data.merge(match_summary[['Match ID', 'Event', 'Match Format', 'Winner', 'Maps Picked/Banned']],
                          on='Match ID', how='left')

# Mapping dictionary for team abbreviations
team_abbreviations = {
    "NRG Esports": "NRG",
    "Furia": "FUR",
    "Cloud9": "C9",
    "MIBR": "MIBR",
    "LOUD": "LOUD",
    "Sentinels": "SEN",
    "LEVIATÁN": "LEV",
    "100 Thieves": "100T",
    "KRÜ Esports": "KRÜ",
    "G2 Esports": "G2",
    "Evil Geniuses": "EG",
    "T1": "T1",
    "BLEED": "BLD",
    "Gen.G": "GEN",
    "Rex Regum Qeon": "RRQ",
    "ZETA DIVISION": "ZETA",
    "Global Esports": "GE",
    "Team Secret": "TS",
    "Talon Esports": "TLN",
    "DRX": "DRX",
    "DetonatioN FocusMe": "DFM",
    "Paper Rex": "PRX",
    "FUT Esports": "FUT",
    "Team Heretics": "TH",
    "GIANTX": "GX",
    "Karmine Corp": "KC",
    "Natus Vincere": "NAVI",
    "BBL Esports": "BBL",
    "Team Liquid": "TL",
    "KOI": "KOI",
    "Team Vitality": "VIT",
    "Gentle Mates": "M8",
    "FNATIC": "FNC",
    "Trace Esports": "TE",
    "TYLOO": "TYL",
    "FunPlus Phoenix": "FPX",
    "Nova Esports": "NOVA",
    "JDG Esports": "JDG",
    "Titan Esports Club": "TEC",
    "Dragon Ranger Gaming": "DRG",
    "All Gamers": "AG",
    "Bilibili Gaming": "BLG",
    "Wolves Esports": "WOL",
    "EDward Gaming": "EDG"
}

# Extract map-specific pick/ban status for each team
# only works for teams that have the same abbreviation as their team name (e.g. DRX)
def get_map_status(row, map_name, team):
    try:
        # Get the team's abbreviation from the dictionary
        team_abbr = team_abbreviations.get(team, team)  # Fallback to team name if not found
        picks_bans = ast.literal_eval(row['Maps Picked/Banned'])
        for action in picks_bans:
            if action['Map'] == map_name:
                if action['Action'] == 'pick' and action['Team'] == team_abbr:
                    return 'Picked'
                elif action['Action'] == 'ban' and action['Team'] == team_abbr:
                    return 'Banned'
                elif action['Action'] == 'remains':
                    return 'Remaining'
    except:
        return None

# Apply map status extraction for both teams
map_data['Team 1 Map Status'] = map_data.apply(lambda row: get_map_status(row, row['Map Name'], row['Team 1']), axis=1)
map_data['Team 2 Map Status'] = map_data.apply(lambda row: get_map_status(row, row['Map Name'], row['Team 2']), axis=1)

# Calculate historical win rates on specific maps for both teams
def calculate_team_win_rate(data, team, map_name, current_date):
    past_matches = data[(data['Date'] < current_date) & 
                        ((data['Team 1'] == team) | (data['Team 2'] == team)) & 
                        (data['Map Name'] == map_name)]
    if len(past_matches) == 0:
        return 0.5  # Default win rate if no history
    wins = sum((past_matches['Team 1'] == team) & (past_matches['Team 1 Score'] > past_matches['Team 2 Score'])) + \
           sum((past_matches['Team 2'] == team) & (past_matches['Team 2 Score'] > past_matches['Team 1 Score']))
    return wins / len(past_matches)

# Apply win rate calculation for both teams
map_data['Team 1 Map Win Rate'] = map_data.apply(lambda row: calculate_team_win_rate(map_data, row['Team 1'], row['Map Name'], row['Date']), axis=1)
map_data['Team 2 Map Win Rate'] = map_data.apply(lambda row: calculate_team_win_rate(map_data, row['Team 2'], row['Map Name'], row['Date']), axis=1)

# Calculate recent form based on average score difference in recent maps (last 3 matches)
def calculate_recent_form(data, team, current_date, num_matches=3):
    # Filter for matches before the current date where the team played
    # Group by Match ID and get the most recent matches
    past_matches = data[(data['Date'] < current_date) & 
                        ((data['Team 1'] == team) | (data['Team 2'] == team))]

    # Group by Match ID and sort by date within each match
    past_matches = past_matches.sort_values(by='Date', ascending=False).drop_duplicates('Match ID', keep='first')
    
    # Select the last 'num_matches' matches
    recent_matches = past_matches.head(num_matches)
    
    if len(recent_matches) == 0:
        return 0  # Default to 0 if no past matches are found

    total_score_diff = 0
    total_maps_played = 0

    # Iterate through the selected matches and calculate the score difference per map
    for match_id in recent_matches['Match ID']:
        # Get all maps played in this match
        match_maps = data[(data['Match ID'] == match_id) & (data['Date'] < current_date)]
        
        # Iterate through each map in the match
        for _, row in match_maps.iterrows():
            score_diff = row['Team 1 Score'] - row['Team 2 Score'] if row['Team 1'] == team else row['Team 2 Score'] - row['Team 1 Score']
            total_score_diff += score_diff
            total_maps_played += 1

    # Calculate the average score difference per map
    if total_maps_played == 0:
        return 0  # Default to 0 if no maps are found

    return total_score_diff / total_maps_played


# Apply recent form calculation for both teams
map_data['Team 1 Recent Form'] = map_data.apply(lambda row: calculate_recent_form(map_data, row['Team 1'], row['Date']), axis=1)
map_data['Team 2 Recent Form'] = map_data.apply(lambda row: calculate_recent_form(map_data, row['Team 2'], row['Date']), axis=1)

# Calculate score difference as the target variable and drop the individual scores
map_data['Score Difference'] = map_data['Team 1 Score'] - map_data['Team 2 Score']
map_data = map_data.drop(columns=['Team 1 Score', 'Team 2 Score'])

# Save the processed data to a new CSV file
map_data.to_csv("processed_map_data_2.csv", index=False)
