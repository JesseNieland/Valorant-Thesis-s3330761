import pandas as pd

df = pd.read_csv("map_level_data_manual copy.csv")
match_level_data = pd.read_csv("match_level_data_manual copy.csv")

# Only include complete matches
match_level_data = match_level_data[match_level_data["Veto Status"] == "Complete"]
df = df[df["Match ID"].isin(match_level_data["Match ID"])]

# Determine halftime scores based on Side Picker and Side Picked
def calculate_halftime_scores(row):
    if row['Side Picker'] == row['Team 1']:
        team1_halftime_score = row['Team 1 Attack Rounds'] if row['Side Picked'] == 'Attack' else row['Team 1 Defense Rounds']
        team2_halftime_score = row['Team 2 Defense Rounds'] if row['Side Picked'] == 'Attack' else row['Team 2 Attack Rounds']
    else:
        team2_halftime_score = row['Team 2 Attack Rounds'] if row['Side Picked'] == 'Attack' else row['Team 2 Defense Rounds']
        team1_halftime_score = row['Team 1 Defense Rounds'] if row['Side Picked'] == 'Attack' else row['Team 1 Attack Rounds']
    return team1_halftime_score, team2_halftime_score
        
df[['Team 1 Halftime Score', 'Team 2 Halftime Score']] = df.apply(
    lambda row: pd.Series(calculate_halftime_scores(row)), axis=1
)

df['Halftime Lead'] = df['Team 1 Halftime Score'] - df['Team 2 Halftime Score']



# Ensure Team Y is always the side picker
def ensure_side_picker(row):
    if row["Side Picker"] == row["Team 1"]:  # If Team 1 is the side picker
        # Swap Team 1 (X) and Team 2 (Y)
        row["Team 1"], row["Team 2"] = row["Team 2"], row["Team 1"]

        # Swap corresponding scores and stats
        row["Team 1 Score"], row["Team 2 Score"] = row["Team 2 Score"], row["Team 1 Score"]
        row["Team 1 Attack Rounds"], row["Team 2 Attack Rounds"] = row["Team 2 Attack Rounds"], row["Team 1 Attack Rounds"]
        row["Team 1 Defense Rounds"], row["Team 2 Defense Rounds"] = row["Team 2 Defense Rounds"], row["Team 1 Defense Rounds"]
        row["Team 1 Overtime Rounds"], row["Team 2 Overtime Rounds"] = row["Team 2 Overtime Rounds"], row["Team 1 Overtime Rounds"]
        row["Team 1 Halftime Score"], row["Team 2 Halftime Score"] = row["Team 2 Halftime Score"], row["Team 1 Halftime Score"]

        # Recalculate derived columns
        row["Score Difference"] = row["Team 1 Score"] - row["Team 2 Score"]
        row["Halftime Lead"] = row["Team 1 Halftime Score"] - row["Team 2 Halftime Score"]
    
    return row

df = df.apply(ensure_side_picker, axis=1)

# Rename columns to Team X and Team Y
df = df.rename(
    columns={
        "Team 1": "Team X",
        "Team 2": "Team Y",
        "Team 1 Score": "Team X Score",
        "Team 2 Score": "Team Y Score",
        "Team 1 Attack Rounds": "Team X Attack Rounds",
        "Team 2 Attack Rounds": "Team Y Attack Rounds",
        "Team 1 Defense Rounds": "Team X Defense Rounds",
        "Team 2 Defense Rounds": "Team Y Defense Rounds",
        "Team 1 Overtime Rounds": "Team X Overtime Rounds",
        "Team 2 Overtime Rounds": "Team Y Overtime Rounds",
        "Team 1 Halftime Score": "Team X Halftime Score",
        "Team 2 Halftime Score": "Team Y Halftime Score",
    }
)

df.to_csv("map_level_data_prepared.csv", index=False)
