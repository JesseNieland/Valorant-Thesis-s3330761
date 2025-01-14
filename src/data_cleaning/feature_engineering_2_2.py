# DOESNT WORK, DISCONTINUED

import pandas as pd
import ast  # To parse the "Maps Picked/Banned" column

# Load the datasets
map_level_data = pd.read_csv("map_data_chrono_3.csv")
match_level_data = pd.read_csv("match_summary_chronological.csv")

# Select necessary columns
match_level_veto_data = match_level_data[['Match ID', 'Maps Picked/Banned']]
merged_data = pd.merge(map_level_data, match_level_veto_data, on="Match ID", how="left")

# Initialize a dictionary to store historical preferences
historical_preferences = {}

# Function to update historical preferences
def update_preferences(team, map_name, action):
    if team not in historical_preferences:
        historical_preferences[team] = {m: {'pick': 0, 'ban': 0, 'total': 0} for m in merged_data['Map Name'].unique()}
    
    if map_name in historical_preferences[team]:
        if action == 'pick':
            historical_preferences[team][map_name]['pick'] += 1
            historical_preferences[team][map_name]['total'] += 1
        elif action == 'ban':
            historical_preferences[team][map_name]['ban'] += 1
            historical_preferences[team][map_name]['total'] += 1

# Function to calculate map preference metrics
def calculate_map_preference(row):
    team_1 = row['Team 1']
    team_2 = row['Team 2']
    map_name = row['Map Name']
    veto_data = row['Maps Picked/Banned']

    # Parse veto actions
    veto_list = ast.literal_eval(veto_data)
    
    # Update historical preferences for the current match
    for action in veto_list:
        team = action['Team']
        map_vetoed = action['Map']
        action_type = action['Action']
        update_preferences(team, map_vetoed, action_type)

    # Calculate preference scores for both teams
    team_1_data = historical_preferences.get(team_1, {}).get(map_name, {})
    team_2_data = historical_preferences.get(team_2, {}).get(map_name, {})

    team_1_preference_score = team_1_data.get('pick', 0) - team_1_data.get('ban', 0)
    team_2_preference_score = team_2_data.get('pick', 0) - team_2_data.get('ban', 0)

    team_1_pick_rate = team_1_data.get('pick', 0) / max(1, team_1_data.get('total', 0))  # Avoid division by zero
    team_2_pick_rate = team_2_data.get('pick', 0) / max(1, team_2_data.get('total', 0))

    team_1_ban_rate = team_1_data.get('ban', 0) / max(1, team_1_data.get('total', 0))
    team_2_ban_rate = team_2_data.get('ban', 0) / max(1, team_2_data.get('total', 0))

    # Return the preference scores and rates
    return pd.Series([team_1_preference_score, team_2_preference_score,
                      team_1_pick_rate, team_2_pick_rate,
                      team_1_ban_rate, team_2_ban_rate])

# Apply the function to calculate historical preference metrics
merged_data[['Team 1 Preference Score', 'Team 2 Preference Score',
             'Team 1 Pick Rate', 'Team 2 Pick Rate',
             'Team 1 Ban Rate', 'Team 2 Ban Rate']] = merged_data.apply(calculate_map_preference, axis=1)

# Function to assign "pick_team_1", "pick_team_2", and "remaining_map" based on "Maps Picked/Banned"
def assign_picks(row):
    veto_data = ast.literal_eval(row['Maps Picked/Banned'])
    map_name = row['Map Name']

    for action in veto_data:
        if action['Map'] == map_name and action['Action'] == 'pick':
            if action['Team'] == row['Team 1']:
                return pd.Series([1, 0, 0])
            elif action['Team'] == row['Team 2']:
                return pd.Series([0, 1, 0])
    return pd.Series([0, 0, 1])  # If neither team explicitly picked the map, it's remaining

# Apply the function to create binary pick indicators
merged_data[['pick_team_1', 'pick_team_2', 'remaining_map']] = merged_data.apply(assign_picks, axis=1)

merged_data = merged_data.drop(columns=['Maps Picked/Banned'])

# Save the updated dataset
merged_data.to_csv("map_data_chrono_3_2_2.csv", index=False)

# Display the first few rows to verify
print(merged_data[['Match ID', 'Map Name', 'Team 1', 'Team 2',
                   'Team 1 Preference Score', 'Team 2 Preference Score',
                   'Team 1 Pick Rate', 'Team 2 Pick Rate',
                   'Team 1 Ban Rate', 'Team 2 Ban Rate']].head(10))
