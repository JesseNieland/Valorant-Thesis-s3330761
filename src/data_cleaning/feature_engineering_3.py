import pandas as pd

# Load dataset
map_data = pd.read_csv("map_data_chrono_3_2.csv")

map_data['Recent Form Difference'] = map_data['Team 1 Recent Form'] - map_data['Team 2 Recent Form']
map_data['Win Rate Difference'] = map_data['Team 1 Win Rate'] - map_data['Team 2 Win Rate']

# Dictionary to store cumulative stats for team-map-side combinations
team_map_stats = {}

# Iterate through the dataset row by row
for idx, row in map_data.iterrows():
    # Get key details
    team_1 = row['Team 1']
    team_2 = row['Team 2']
    map_name = row['Map Name']
    team_1_attack = row['Team 1 Attack Rounds']
    team_1_defense = row['Team 1 Defense Rounds']
    team_2_attack = row['Team 2 Attack Rounds']
    team_2_defense = row['Team 2 Defense Rounds']

    # Initialize keys for each team, map, and side
    key_team_1_attack = (team_1, map_name, 'attack')
    key_team_1_defense = (team_1, map_name, 'defense')
    key_team_2_attack = (team_2, map_name, 'attack')
    key_team_2_defense = (team_2, map_name, 'defense')

    # Calculate and assign win rates for Team 1
    for key, rounds_won, rounds_played in [
        (key_team_1_attack, team_1_attack, team_1_attack + team_2_defense),
        (key_team_1_defense, team_1_defense, team_1_defense + team_2_attack),
    ]:
        if key in team_map_stats:
            cumulative_stats = team_map_stats[key]
            win_rate = cumulative_stats['rounds_won'] / cumulative_stats['rounds_played']
        else:
            win_rate = 0.5  # Default win rate for new combinations
        map_data.loc[idx, f"Team 1 {key[2].capitalize()} Win Rate"] = win_rate

    # Calculate and assign win rates for Team 2
    for key, rounds_won, rounds_played in [
        (key_team_2_attack, team_2_attack, team_2_attack + team_1_defense),
        (key_team_2_defense, team_2_defense, team_2_defense + team_1_attack),
    ]:
        if key in team_map_stats:
            cumulative_stats = team_map_stats[key]
            win_rate = cumulative_stats['rounds_won'] / cumulative_stats['rounds_played']
        else:
            win_rate = 0.5  # Default win rate for new combinations
        map_data.loc[idx, f"Team 2 {key[2].capitalize()} Win Rate"] = win_rate

    # Update cumulative stats for Team 1
    for key, rounds_won, rounds_played in [
        (key_team_1_attack, team_1_attack, team_1_attack + team_2_defense),
        (key_team_1_defense, team_1_defense, team_1_defense + team_2_attack),
    ]:
        if key not in team_map_stats:
            team_map_stats[key] = {'rounds_won': 0, 'rounds_played': 0}
        team_map_stats[key]['rounds_won'] += rounds_won
        team_map_stats[key]['rounds_played'] += rounds_played

    # Update cumulative stats for Team 2
    for key, rounds_won, rounds_played in [
        (key_team_2_attack, team_2_attack, team_2_attack + team_1_defense),
        (key_team_2_defense, team_2_defense, team_2_defense + team_1_attack),
    ]:
        if key not in team_map_stats:
            team_map_stats[key] = {'rounds_won': 0, 'rounds_played': 0}
        team_map_stats[key]['rounds_won'] += rounds_won
        team_map_stats[key]['rounds_played'] += rounds_played


# Initialize a cumulative stats dictionary for rounds played on each map and side
team_performance_stats = {}

# Function to update stats and calculate differences based on rounds played (not won)
def calculate_round_differences(data):
    attack_round_diff = []
    defense_round_diff = []

    for _, row in data.iterrows():
        team_1 = row['Team 1']
        team_2 = row['Team 2']
        map_name = row['Map Name']
        
        # Create keys for the map-specific stats
        team_1_key = (team_1, map_name)
        team_2_key = (team_2, map_name)

        # Initialize stats for teams if not present
        if team_1_key not in team_performance_stats:
            team_performance_stats[team_1_key] = {'attack_rounds_played': 0, 'defense_rounds_played': 0}
        if team_2_key not in team_performance_stats:
            team_performance_stats[team_2_key] = {'attack_rounds_played': 0, 'defense_rounds_played': 0}

        # Calculate differences based on current cumulative rounds played (not won)
        team_1_attack_diff = (
            team_performance_stats[team_1_key]['attack_rounds_played'] - team_performance_stats[team_2_key]['attack_rounds_played']
        )
        team_1_defense_diff = (
            team_performance_stats[team_1_key]['defense_rounds_played'] - team_performance_stats[team_2_key]['defense_rounds_played']
        )

        # Append calculated differences
        attack_round_diff.append(team_1_attack_diff)
        defense_round_diff.append(team_1_defense_diff)

        # Update cumulative stats for the current match based on rounds played
        team_performance_stats[team_1_key]['attack_rounds_played'] += (row['Team 1 Attack Rounds'] + row['Team 2 Defense Rounds'])
        team_performance_stats[team_1_key]['defense_rounds_played'] += (row['Team 1 Defense Rounds'] + row['Team 2 Attack Rounds'])
        team_performance_stats[team_2_key]['attack_rounds_played'] += (row['Team 2 Attack Rounds'] + row['Team 1 Defense Rounds'])
        team_performance_stats[team_2_key]['defense_rounds_played'] += (row['Team 2 Defense Rounds'] + row['Team 1 Attack Rounds'])

    # Add new columns to the dataframe
    data['Attack Rounds Played Difference'] = attack_round_diff
    data['Defense Rounds Played Difference'] = defense_round_diff
    return data

# Apply the function to the dataset
updated_data = calculate_round_differences(map_data)



# Categorical columns drop
map_data.drop(columns=['Match ID', 'Date'], inplace=True)
# Recent form and Win rate drop
map_data.drop(columns=['Team 1 Recent Form','Team 2 Recent Form','Team 1 Win Rate','Team 2 Win Rate'], inplace=True)
# Scores per Side drop
map_data.drop(columns=['Team 1 Attack Rounds','Team 1 Defense Rounds','Team 2 Attack Rounds','Team 2 Defense Rounds'], inplace=True)

# Check for any null values
map_data.isnull().sum()

# Save the updated dataset
map_data.to_csv("map_data_chrono_3_3_1.csv", index=False)
