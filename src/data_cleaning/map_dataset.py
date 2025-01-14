import pandas as pd

# Load the dataset
df = pd.read_csv("vct_2024_matches_formatted.csv")

df['Match ID'] = range(1, len(df) + 1)

# Initialize an empty list to store map-level data
map_data = []

# Loop through each row in the match-level dataset
for i, row in df.iterrows():
    match_id = row['Match ID']
    team_1 = row['Team 1']
    team_2 = row['Team 2']
    date = row['Date']
    
    # Parse Map Scores and other map-specific columns
    map_scores = eval(row['Map Scores'])  # List of dictionaries for each map's score
    team_1_attack = eval(row['Team 1 Attack'])  # List of attack rounds for Team 1
    team_1_defense = eval(row['Team 1 Defense'])  # List of defense rounds for Team 1
    team_2_attack = eval(row['Team 2 Attack'])  # List of attack rounds for Team 2
    team_2_defense = eval(row['Team 2 Defense'])  # List of defense rounds for Team 2
    durations = eval(row['Durations (in seconds)'])  # List of map durations

    # Loop through each map in the match
    for j, map_score in enumerate(map_scores):
        # Extract map-specific data
        map_name = map_score['Map']
        team_1_score = map_score['Team 1 Score']
        team_2_score = map_score['Team 2 Score']
        
        # Handle missing values in lists using `None` if index is out of range
        map_duration = durations[j] if j < len(durations) else None
        team_1_attack_rounds = team_1_attack[j] if j < len(team_1_attack) else None
        team_1_defense_rounds = team_1_defense[j] if j < len(team_1_defense) else None
        team_2_attack_rounds = team_2_attack[j] if j < len(team_2_attack) else None
        team_2_defense_rounds = team_2_defense[j] if j < len(team_2_defense) else None

        # Append map-level data to the list
        map_data.append({
            'Match ID': match_id,
            'Date': date,
            'Team 1': team_1,
            'Team 2': team_2,
            'Map Name': map_name,
            'Map Duration': map_duration,
            'Team 1 Score': team_1_score,
            'Team 2 Score': team_2_score,
            'Team 1 Attack Rounds': team_1_attack_rounds,
            'Team 1 Defense Rounds': team_1_defense_rounds,
            'Team 2 Attack Rounds': team_2_attack_rounds,
            'Team 2 Defense Rounds': team_2_defense_rounds
        })

# Convert the list of dictionaries into a DataFrame
map_df = pd.DataFrame(map_data)

# Display the map-level dataset
print(map_df.head())

# Save the map-level dataset to a CSV file
map_df.to_csv("map_level_dataset.csv", index=False)
