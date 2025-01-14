import pandas as pd

# Load dataset
map_data = pd.read_csv("map_data_chrono_2_2.csv")

# Feature: Recent Form Based on Last 3 Matches
def calculate_recent_form_matches(data, team, current_date, num_matches=3):
    # Filter for matches before the current date where the team played
    past_matches = data[(data['Date'] < current_date) & 
                        ((data['Team 1'] == team) | (data['Team 2'] == team))]

    # Sort by Date and drop duplicate Match IDs to get one row per match
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
            # Use 'Score Difference' column to get score difference
            score_diff = row['Score Difference'] if row['Team 1'] == team else -row['Score Difference']
            total_score_diff += score_diff
            total_maps_played += 1

    # Calculate the average score difference per map
    if total_maps_played == 0:
        return 0  # Default to 0 if no maps are found

    return total_score_diff / total_maps_played


# Apply recent form calculation for both teams based on matches
map_data['Team 1 Recent Form'] = map_data.apply(
    lambda row: calculate_recent_form_matches(map_data, row['Team 1'], row['Date']), axis=1)
map_data['Team 2 Recent Form'] = map_data.apply(
    lambda row: calculate_recent_form_matches(map_data, row['Team 2'], row['Date']), axis=1)

# Save the updated dataset
map_data.to_csv("map_data_with_recent_form_matches.csv", index=False)


# Feature 2: Overall Win Rate
def calculate_overall_win_rate(data, team, current_date):
    # Filter for matches before the current date where the team played
    past_matches = data[(data['Date'] < current_date) & 
                        ((data['Team 1'] == team) | (data['Team 2'] == team))]
    
    if len(past_matches) == 0:
        return 0.5  # Default win rate if no history

    # Calculate win count for the team
    wins = sum((past_matches['Team 1'] == team) & (past_matches['Score Difference'] > 0)) + \
           sum((past_matches['Team 2'] == team) & (past_matches['Score Difference'] < 0))
    return wins / len(past_matches)

# Apply win rate calculation for both teams
map_data['Team 1 Win Rate'] = map_data.apply(
    lambda row: calculate_overall_win_rate(map_data, row['Team 1'], row['Date']), axis=1)
map_data['Team 2 Win Rate'] = map_data.apply(
    lambda row: calculate_overall_win_rate(map_data, row['Team 2'], row['Date']), axis=1)

# Save the updated dataset
map_data.to_csv("map_data_chrono_3.csv", index=False)
