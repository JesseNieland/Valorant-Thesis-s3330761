import pandas as pd

# Load dataset
df = pd.read_csv("vct_2024_matches_formatted.csv")

df['Match ID'] = range(1, len(df) + 1)

# Initialize lists to store results
total_maps_played = []
match_winner = []

# Iterate over each row to calculate total maps and winner
for i, row in df.iterrows():
    map_scores = eval(row['Map Scores'])  # Parse map scores
    team_1_wins = 0
    team_2_wins = 0

    # Calculate map wins for each team
    for map_score in map_scores:
        team_1_score = map_score['Team 1 Score']
        team_2_score = map_score['Team 2 Score']
        if team_1_score > team_2_score:
            team_1_wins += 1
        elif team_2_score > team_1_score:
            team_2_wins += 1

    # Total maps played
    total_maps_played.append(len(map_scores))

    # Determine match winner
    if team_1_wins > team_2_wins:
        match_winner.append(row['Team 1'])
    elif team_2_wins > team_1_wins:
        match_winner.append(row['Team 2'])
    else:
        match_winner.append("Tie")  # Optional: handle ties if applicable

# Add columns to DataFrame
df['Total Maps Played'] = total_maps_played
df['Winner'] = match_winner

# Select columns for the summary dataset
summary_df = df[['Match ID', 'Event', 'Team 1', 'Team 2', 'Date', 'Match Format', 'Maps Picked/Banned', 'Total Maps Played', 'Winner']]

# Save the updated summary dataset
summary_df.to_csv("match_summary_dataset_with_winner.csv", index=False)
