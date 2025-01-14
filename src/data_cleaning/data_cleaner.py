import pandas as pd
import ast
import re

# Load the dataset
df = pd.read_csv("vct_2024_matches_cleaned_final.csv")

# Function to clean "Maps Picked/Banned" column
def clean_picked_banned(maps_str):
    actions = maps_str.split(';')
    structured_actions = []
    for action in actions:
        match = re.match(r"(\w+) (ban|pick) (\w+)", action.strip())
        if match:
            team, action_type, map_name = match.groups()
            structured_actions.append({"Team": team, "Action": action_type, "Map": map_name})
        elif "remains" in action:
            remaining_map = action.strip().split(" ")[0]
            structured_actions.append({"Team": "N/A", "Action": "remains", "Map": remaining_map})
    return structured_actions

# Apply the function and convert lists to clean string format
df['Maps Picked/Banned'] = df['Maps Picked/Banned'].apply(clean_picked_banned)

# Function to clean "Map Scores" column with a loop to handle multiple maps
def clean_map_scores(scores_str):
    try:
        # Safely evaluate the string as a list
        scores = ast.literal_eval(scores_str)
    except (ValueError, SyntaxError):
        return []
    
    structured_scores = []
    for score in scores:
        map_match = re.search(r"(\w+): (\d+)-(\d+)", score)
        if map_match:
            map_name, team1_score, team2_score = map_match.groups()
            structured_scores.append({
                "Map": map_name,
                "Team 1 Score": int(team1_score),
                "Team 2 Score": int(team2_score)
            })
    return structured_scores

# Apply the function and convert lists to clean string format
df['Map Scores'] = df['Map Scores'].apply(clean_map_scores)

df.to_csv("vct_2024_matches_formatted.csv", index=False)

print("Data saved to vct_2024_matches_formatted.csv")
print(df[['Maps Picked/Banned', 'Map Scores']].head())
