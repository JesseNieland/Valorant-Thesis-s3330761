import pandas as pd
import ast  # To parse the "Maps Picked/Banned" column

# Mapping dictionary for team abbreviations
team_abbreviations = {
    "NRG Esports": "NRG",
    "FURIA": "FUR",
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

# Load datasets
map_level_data = pd.read_csv("map_data_chrono_3.csv")
match_level_data = pd.read_csv('match_summary_chronological.csv')

# Select only the necessary columns from match-level data (i.e., 'Match ID' and 'Maps Picked/Banned')
match_level_veto_data = match_level_data[['Match ID', 'Maps Picked/Banned']]

# Merge the veto data with the map-level data on 'Match ID'
merged_data = pd.merge(map_level_data, match_level_veto_data, on='Match ID', how='left')

# Define the function to calculate map preference with the team abbreviation conversion
def calculate_map_preference(row):
    veto_data = row['Maps Picked/Banned']  # Veto actions from the match level
    map_name = row['Map Name']  # The map being played
    team_1 = row['Team 1']  # Team 1 from the map-level data
    team_2 = row['Team 2']  # Team 2 from the map-level data
    
    # Convert team names to abbreviations using the mapping dictionary
    team_1_abbr = team_abbreviations.get(team_1, team_1)  # If no match, keep original name
    team_2_abbr = team_abbreviations.get(team_2, team_2)  # If no match, keep original name
    
    # Convert the veto data (which is stored as a string) into a list of dictionaries
    veto_list = ast.literal_eval(veto_data)  # Parse the veto data from string to list of dicts
    
    # Track map preferences for each team based on veto actions
    team_1_preference = 0
    team_2_preference = 0
    
    # Loop through the veto actions and determine who picked/banned each map
    for action in veto_list:
        team = action['Team']
        map_vetoed = action['Map']
        action_type = action['Action']
        
        if map_vetoed == map_name:
            if action_type == 'pick':
                if team == team_1_abbr:
                    team_1_preference += 1  # Team 1 picked this map
                elif team == team_2_abbr:
                    team_2_preference += 1  # Team 2 picked this map
            elif action_type == 'ban':
                if team == team_1_abbr:
                    team_1_preference -= 1  # Team 1 banned this map
                elif team == team_2_abbr:
                    team_2_preference -= 1  # Team 2 banned this map
    
    # Compare preferences for Team 1 and Team 2 to determine map preference
    if team_1_preference > team_2_preference:
        return 1  # Team 1 prefers this map
    elif team_2_preference > team_1_preference:
        return -1  # Team 2 prefers this map
    else:
        return 0  # No clear preference

# Apply the function to calculate map preference for each row in the merged dataset
merged_data['Map Preference'] = merged_data.apply(calculate_map_preference, axis=1)

merged_data = merged_data.drop(columns=['Maps Picked/Banned'])

# Function to assign the "pick_team_1", "pick_team_2", and "remaining_map" based on "Map Preference"
def assign_picks(row):
    if row['Map Preference'] == 1:
        # Team 1 picked the map
        return pd.Series([1, 0, 0])
    elif row['Map Preference'] == -1:
        # Team 2 picked the map
        return pd.Series([0, 1, 0])
    else:
        # The map is remaining (neither team picked it)
        return pd.Series([0, 0, 1])

# Apply the function to create the new columns
merged_data[['pick_team_1', 'pick_team_2', 'remaining_map']] = merged_data.apply(assign_picks, axis=1)

# Drop the "Map Preference" column as it is no longer needed
merged_data.drop(columns=['Map Preference'], inplace=True)


# Print the first few rows to verify the result
print(merged_data[['Match ID', 'Map Name', 'pick_team_1', 'pick_team_2', 'remaining_map']].head(10))

merged_data.to_csv("map_data_chrono_3_2.csv", index=False)
