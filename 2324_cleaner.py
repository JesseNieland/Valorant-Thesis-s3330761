import pandas as pd
import re
import ast

# Load the datasets
data_2023 = pd.read_csv('vct_2023_matches_2_manual.csv')
data_2024 = pd.read_csv('vct_2024_matches_2_manual.csv')

data_2023['Year'] = 2023
data_2024['Year'] = 2024

# Combine the datasets
combined_data = pd.concat([data_2023, data_2024], ignore_index=True)



# Standardizing name mapping
team_name_mapping = {
    "KOI": "KOI",
    "Movistar KOI": "KOI",
    "Giants Gaming": "GIANTX",
    "GIANTX": "GIANTX",
}

def standardize_team_name(team_name):
    """
    Standardizes team names based on defined mapping
    """
    for key, value in team_name_mapping.items():
        if key in team_name:
            return value
    return team_name

combined_data['Team 1'] = combined_data['Team 1'].apply(standardize_team_name)
combined_data['Team 2'] = combined_data['Team 2'].apply(standardize_team_name)



# Order data by event start date, rows ordered per event
# Ensure the 'Date' column is in datetime format
combined_data['Date'] = pd.to_datetime(combined_data['Date'], errors='coerce')

# Get the earliest date for each event to determine chronological event order
event_start_dates = combined_data.groupby('Event')['Date'].min().reset_index()
event_start_dates = event_start_dates.sort_values(by='Date', ascending=True).reset_index(drop=True)

# Add a chronological rank to each event based on the start date
event_start_dates['Event_Rank'] = range(1, len(event_start_dates) + 1)
combined_data = combined_data.merge(event_start_dates[['Event', 'Event_Rank']], on='Event', how='left')
combined_data = combined_data.sort_values(by=['Event_Rank', 'Date']).reset_index(drop=True)

# Assign new Match IDs in the order of the rows in the combined dataset
combined_data['Match_ID'] = range(1, len(combined_data) + 1)
combined_data = combined_data[['Match_ID'] + [col for col in combined_data.columns if col != 'Match_ID']]

combined_data = combined_data.drop(columns=['Event_Rank'])


# Adding a Veto Status column that marks maps/rows with missing Map Veto data
combined_data['Veto Status'] = combined_data['Maps Picked/Banned'].apply(lambda x: 'Missing' if 'No map veto information available' in str(x) else 'Complete')



def clean_picked_banned(maps_str):
    """
    Change Maps Picked/Banned to be properly formatted.
    """

    if maps_str in [None, "", []]:  # Handle missing or empty data
        return []
    actions = maps_str.split(';')
    structured_actions = []
    for action in actions:
        match = re.match(r"(\w+) (ban|pick) (\w+)", action.strip()) # Team_Name ban/pick Map_Name 
        if match:
            team, action_type, map_name = match.groups()
            structured_actions.append({"Team": team, "Action": action_type, "Map": map_name})
        elif "remains" in action:
            remaining_map = action.strip().split(" ")[0]
            structured_actions.append({"Team": "N/A", "Action": "remains", "Map": remaining_map})
    return structured_actions

# Update "Maps Picked/Banned" for rows with missing veto data
combined_data.loc[combined_data['Veto Status'] == 'Missing', 'Maps Picked/Banned'] = None

combined_data['Maps Picked/Banned'] = combined_data['Maps Picked/Banned'].apply(clean_picked_banned)



# Mapping dictionary for team abbreviations
# Number One Player and Kingzone do not have abbreviations according to the data (added placeholders)
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
    "TALON": "TLN",
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
    "EDward Gaming": "EDG",
    "Monarch Effect": "ME",
    "Weibo Gaming": "WBG",
    "Four Angry Men": "4AM",
    "Shenzhen NTER": "NTER",
    "Gank Gaming": "GK",
    "Night Wings Gaming": "NWG",
    "Attacking Soul Esports": "ASE",
    "Douyu Gaming": "DYG",
    "Royal Never Give Up": "RNG",
    "Number One Player": "NOP",
    "Invincible Gaming": "iNv",
    "Totoro Gaming": "TTG",
    "Rare Atom": "RA",
    "Kingzone": "KZ"
}

# Reverse mapping for abbreviations to full names
abbreviation_to_team = {v: k for k, v in team_abbreviations.items()}

def map_abbreviations_to_full_names(maps_list):
    """
    Replace abbreviations with full team names in the Maps Picked/Banned column.
    """
    if not maps_list or maps_list == []:  # Handle missing data
        return []
    updated_list = []
    for action in maps_list:
        team_abbreviation = action["Team"]
        if team_abbreviation == "GIA":
            team_abbreviation = "GX"
        full_team_name = abbreviation_to_team.get(team_abbreviation, team_abbreviation)
        action["Team"] = full_team_name  # Replace abbreviation with full name
        updated_list.append(action)
    return updated_list

combined_data['Maps Picked/Banned'] = combined_data['Maps Picked/Banned'].apply(map_abbreviations_to_full_names)



def assign_team_roles(map_veto):
    """
    Determines Team A and Team B based on the Map Veto.
    Team A is the team that picks the first map.
    Team B is the team that picks the second map.
    """
    if not map_veto or not isinstance(map_veto, list):
        # Handle missing or invalid map veto data
        return {"Team A": None, "Team B": None}
    
    team_a = None
    team_b = None

    # Loop through the veto actions to find the first two picks
    pick_count = 0
    for action in map_veto:
        if action["Action"] == "pick":
            pick_count += 1
            if pick_count == 1:
                # First pick defines Team A
                team_a = action["Team"]
            elif pick_count == 2:
                # Second pick defines Team B
                team_b = action["Team"]
                break  # Exit loop early once both teams are identified

    return {"Team A": team_a, "Team B": team_b}

combined_data[['Team A', 'Team B']] = combined_data.apply(
    lambda row: pd.Series(assign_team_roles(row['Maps Picked/Banned'])),
    axis=1
)



# Clean the the structure of the dataset
# Drop unneccesary columns
combined_data = combined_data.drop(columns=['Match Durations', 'Match URL'])

# Rearrange the order of columns
column_order = ['Match_ID', 'Event', 'Year', 'Date', 'Match Format', 'Team 1', 'Team 2', 'Team A', 'Team B', 'Maps Picked/Banned', 
                'Score per Map', 'Score per Half', 'Sides Chosen by Non-Picker', 'Veto Status']
combined_data = combined_data[column_order]

# Rename columns
combined_data.rename(columns={
    'Match_ID': 'Match ID',
    'Maps Picked/Banned': 'Map Veto',
    'Score per Map': 'Map Scores',
    'Score per Half': 'Half Scores',
    'Sides Chosen by Non-Picker': 'Starting Sides'
}, inplace=True)



# # Save the combined dataset
# combined_data.to_csv('combined_matches_2023_2024_2_manual.csv', index=False)

# # Display a snippet of the combined data
# print(combined_data.head())



def calculate_maps_played(map_scores):
    """
    Calculates the number of maps played in a match.
    """
    if not map_scores or pd.isna(map_scores):  # Handle missing or NaN values
        return 0
    return len(map_scores.split(';'))  # Count the number of maps by splitting on ';'

combined_data['Maps Played'] = combined_data['Map Scores'].apply(calculate_maps_played)



def parse_map_level_data(row):
    """
    Parses map-level data from the row into a structured format.
    Makes sure overtime rounds are integers and default to 0 if missing.
    """
    try:
        map_veto = ast.literal_eval(row["Map Veto"])  # Convert string to list of dictionaries
    except (ValueError, SyntaxError):
        map_veto = []
    
    map_scores = row["Map Scores"].split("; ") if pd.notna(row["Map Scores"]) else []
    half_scores = row["Half Scores"].split("; ") if pd.notna(row["Half Scores"]) else []
    
    # Set data lengths
    num_maps = max(len(map_veto), len(map_scores), len(half_scores))
    map_veto += [None] * (num_maps - len(map_veto))
    map_scores += [None] * (num_maps - len(map_scores))
    half_scores += [None] * (num_maps - len(half_scores))
    
    # Parse map-level rows
    map_rows = []
    for i in range(num_maps):
        # Extract map name
        map_name = None
        if map_veto[i] and "Map" in map_veto[i]:
            map_name = map_veto[i]["Map"]
        elif map_scores[i]:
            match = re.match(r"^(.*?):", map_scores[i]) # Gets everything before the :
            if match:
                map_name = match.group(1)
        
        team1_score, team2_score = None, None
        team1_attack, team1_defense, team1_ot = None, None, 0
        team2_defense, team2_attack, team2_ot = None, None, 0
        
        # Parse scores
        if map_scores[i]:
            try:
                team1_score, team2_score = map(int, map_scores[i].split(":")[1].split("-"))
            except (ValueError, IndexError):
                pass
        
        # Parse half scores
        if half_scores[i]:
            try:
                parts = half_scores[i].split(", ")
                for part in parts:
                    if "Team 1" in part:
                        scores = part.split(": ")[1].split(" / ")
                        team1_attack, team1_defense = map(int, scores[:2])
                        if len(scores) > 2:
                            team1_ot = int(float(scores[2]))  # Convert OT rounds to integer
                    elif "Team 2" in part:
                        scores = part.split(": ")[1].split(" / ")
                        team2_defense, team2_attack = map(int, scores[:2])
                        if len(scores) > 2:
                            team2_ot = int(float(scores[2]))  # Convert OT rounds to integer
            except (ValueError, IndexError):
                pass
        
        # Append row
        map_rows.append({
            "Match ID": row["Match ID"],
            "Team 1": row["Team 1"],
            "Team 2": row["Team 2"],
            "Team A": row["Team A"],
            "Team B": row["Team B"],
            "Map Name": map_name,
            "Team 1 Score": team1_score,
            "Team 2 Score": team2_score,
            "Team 1 Attack Rounds": team1_attack,
            "Team 1 Defense Rounds": team1_defense,
            "Team 1 Overtime Rounds": team1_ot, 
            "Team 2 Attack Rounds": team2_attack,
            "Team 2 Defense Rounds": team2_defense,
            "Team 2 Overtime Rounds": team2_ot,
        })
    
    return map_rows

# Create Map-Level Dataset
map_level_data = combined_data.apply(parse_map_level_data, axis=1).explode().reset_index(drop=True)
map_level_data = pd.json_normalize(map_level_data)

# Create Match-Level Dataset
match_level_columns = [
    "Match ID", "Event", "Year", "Date", "Match Format",
    "Team 1", "Team 2", "Team A", "Team B", "Map Veto", "Starting Sides", "Veto Status", "Maps Played"
]
match_level_data = combined_data[match_level_columns]



def determine_side_picker(map_row):
    """
    Determines which team picks the side for a given map.
    """
    match_id = map_row["Match ID"]
    map_index = map_row["Map Index"]  # Index of the map in the match
    
    # Retrieve the match format and Team A/B from match data
    match_row = match_level_data.loc[match_level_data["Match ID"] == match_id]
    if match_row.empty:
        return None
    
    match_format = match_row.iloc[0]["Match Format"]
    team_a = match_row.iloc[0]["Team A"]
    team_b = match_row.iloc[0]["Team B"]
    
    # Determine the side picker based on match format
    if match_format == "Bo3":
        if map_index == 1:
            return team_b  # Map 1: Team B picks side
        elif map_index in [2, 3]:
            return team_a  # Map 2, 3: Team A picks side
    elif match_format == "Bo5":
        if map_index in [1, 3, 5]:
            return team_b  # Maps 1, 3, 5: Team B picks side
        elif map_index in [2, 4]:
            return team_a  # Maps 2, 4: Team A picks side
    return None  # For any unexpected case

# Add a column for map index in the map level dataset
map_level_data["Map Index"] = map_level_data.groupby("Match ID").cumcount() + 1

map_level_data["Side Picker"] = map_level_data.apply(determine_side_picker, axis=1)



def determine_side_picked(row):
    """
    Determines which side was picked for each map based on the Starting Sides, Side Picker,
    Team 1, Team 2, and Match Format.
    """
    match_id = row["Match ID"]
    map_index = row["Map Index"]  # Index of the map in the match
    
    # Get match-level data
    match_row = match_level_data.loc[match_level_data["Match ID"] == match_id]
    if match_row.empty:
        return None

    starting_sides = match_row.iloc[0]["Starting Sides"]
    side_picker = row["Side Picker"]
    team1 = row["Team 1"]
    team2 = row["Team 2"]

    # Parse the Starting Sides column for the given map
    sides_per_map = starting_sides.split(";")
    if map_index > len(sides_per_map):
        return None  # Map index exceeds available data

    map_sides = sides_per_map[map_index - 1].strip()  # Extract the relevant map's sides

    # Handle cases where both sides are provided with "&" (e.g., "Team 1: Attack & Team 2: Defense")
    if "&" in map_sides:
        team1_side, team2_side = map_sides.split("&")
        team1_side = team1_side.strip()
        team2_side = team2_side.strip()

        if side_picker == team1:
            return team1_side.split(":")[1].strip()  # Extract and return the side picked by Team 1
        elif side_picker == team2:
            return team2_side.split(":")[1].strip()  # Extract and return the side picked by Team 2
    else:
        # Handle single-team side information (e.g., "Team 1: Attack")
        if "Team 1" in map_sides:
            chosen_side = map_sides.split(":")[1].strip()  # Extract side (Attack/Defense)
            if side_picker == team1:
                return chosen_side
        elif "Team 2" in map_sides:
            chosen_side = map_sides.split(":")[1].strip()  # Extract side (Attack/Defense)
            if side_picker == team2:
                return chosen_side

    return None  # Default case if no side can be determined

map_level_data["Side Picked"] = map_level_data.apply(determine_side_picked, axis=1)



# Add an 'Overtime' column to the map-level dataset
map_level_data['Overtime'] = (
    (map_level_data['Team 1 Overtime Rounds'] > 0) |
    (map_level_data['Team 2 Overtime Rounds'] > 0) |
    (map_level_data['Team 1 Score'] > 13) |
    (map_level_data['Team 2 Score'] > 13)
).astype(int)



def determine_map_winner(row):
    """
    Determine the winner of the map.
    """
    if row['Team 1 Score'] > row['Team 2 Score']:
        return row['Team 1']
    elif row['Team 1 Score'] < row['Team 2 Score']:
        return row['Team 2']
    else:
        return None  # Handle ties or errors (if any)

map_level_data['Map Winner'] = map_level_data.apply(determine_map_winner, axis=1)



def determine_match_winner(match_id):
    """
    Determine the winner of the match.
    """
    maps_in_match = map_level_data[map_level_data['Match ID'] == match_id]
    team1_wins = (maps_in_match['Map Winner'] == maps_in_match['Team 1']).sum()
    team2_wins = (maps_in_match['Map Winner'] == maps_in_match['Team 2']).sum()
    
    if team1_wins > team2_wins:
        return maps_in_match['Team 1'].iloc[0]  # Return Team 1
    elif team2_wins > team1_wins:
        return maps_in_match['Team 2'].iloc[0]  # Return Team 2
    else:
        return None  # Handle ties or errors (if any)

match_level_data['Match Winner'] = match_level_data['Match ID'].map(determine_match_winner)



# Create a Score Difference column for the map level dataset
map_level_data["Score Difference"] = map_level_data["Team 1 Score"] - map_level_data["Team 2 Score"]

# Add a Total Rounds Played column to the map-level dataset
map_level_data['Total Rounds Played'] = map_level_data['Team 1 Score'] + map_level_data['Team 2 Score']



def extract_map_pool(veto_data):
    """
    Extract a sorted list of unique maps from the Map Veto column.
    """
    try:
        # Check if the data is already a list
        if isinstance(veto_data, str):
            veto_list = ast.literal_eval(veto_data)  # Parse the string
        elif isinstance(veto_data, list):
            veto_list = veto_data
        else:
            return None  # Handle unexpected formats  
        
        # Extract the 'Map' values
        maps = {action["Map"] for action in veto_list if "Map" in action}
        
        # Return a sorted list of unique maps
        return sorted(maps)
    except (ValueError, SyntaxError):
        return None
    except KeyError:
        return None

match_level_data['Map Pool'] = match_level_data['Map Veto'].apply(extract_map_pool)

# Define the map pool to assign for empty lists (China Champions 2023 Qualifier)
china_qualifiers_map_pool = ['Ascent', 'Bind', 'Fracture', 'Haven', 'Lotus', 'Pearl', 'Split']

# Replace empty lists in the 'Map Pool' column with the map pool
match_level_data['Map Pool'] = match_level_data['Map Pool'].apply(
    lambda pool: china_qualifiers_map_pool if not pool else pool
)

# Ensure all Map Pool values are tuples
match_level_data['Map Pool'] = match_level_data['Map Pool'].apply(lambda pool: tuple(pool) if isinstance(pool, list) else pool)

# Get unique map pools and assign them a patch number
unique_map_pools = {pool: idx + 1 for idx, pool in enumerate(match_level_data['Map Pool'].dropna().unique())}

# Create a new column to store the patch number
match_level_data['Patch'] = match_level_data['Map Pool'].apply(lambda pool: unique_map_pools[pool])



# Mapping of events in chronological order
event_order = {
    "Champions Tour 2023: LOCK//IN São Paulo": 1,
    "Champions Tour 2023: Americas League": 2,
    "Champions Tour 2023: EMEA League": 2,
    "Champions Tour 2023: Pacific League": 2,
    "Champions Tour 2023: Champions China Qualifier (Preliminary)": 3,
    "Champions Tour 2023: Masters Tokyo": 4,
    "Champions Tour 2023: Champions China Qualifier (Playoffs)": 5,
    "Champions Tour 2023: Americas Last Chance Qualifier": 6,
    "Champions Tour 2023: EMEA Last Chance Qualifier": 6,
    "Champions Tour 2023: Pacific Last Chance Qualifier": 6,
    "Valorant Champions 2023": 7,
    "Champions Tour 2024: Americas Kickoff": 8,
    "Champions Tour 2024: Pacific Kickoff": 8,
    "Champions Tour 2024: EMEA Kickoff": 8,
    "Champions Tour 2024: China Kickoff": 8,
    "Champions Tour 2024: Masters Madrid": 9,
    "Champions Tour 2024: Americas Stage 1": 10,
    "Champions Tour 2024: EMEA Stage 1": 10,
    "Champions Tour 2024: Pacific Stage 1": 10,
    "Champions Tour 2024: China Stage 1": 10,
    "Champions Tour 2024: Masters Shanghai": 11,
    "Champions Tour 2024: Americas Stage 2": 12,
    "Champions Tour 2024: EMEA Stage 2": 12,
    "Champions Tour 2024: Pacific Stage 2": 12,
    "Champions Tour 2024: China Stage 2": 12,
    "Valorant Champions 2024": 13,
}

def assign_chronological_order(row):
    """
    Assigns a chronological order to events.
    Some regional events happen at the same time.
    """
    event_name = row["Event"]
    event_date = row["Date"]
    
    # Handle Chinese Champions Qualifier in 2023
    if event_name == "Champions Tour 2023: Champions China Qualifier":
        if pd.to_datetime(event_date) < pd.to_datetime("2023-06-11"):  # Before Masters Tokyo
            return event_order["Champions Tour 2023: Champions China Qualifier (Preliminary)"]
        else:  # After Masters Tokyo
            return event_order["Champions Tour 2023: Champions China Qualifier (Playoffs)"]
    
    # Default mapping for other events
    return event_order.get(event_name, None)  # None if event not found in mapping

match_level_data["Chronological Order"] = match_level_data.apply(assign_chronological_order, axis=1)



map_level_data = map_level_data.merge(
    match_level_data[["Match ID", "Match Format"]], on="Match ID", how="left"
)

def determine_map_picker(row):
    """
    Determines the map picker.
    """
    # Identify neutral maps based on Match Format and Map Index
    if (row['Match Format'] == 'Bo3' and row['Map Index'] == 3) or \
       (row['Match Format'] == 'Bo5' and row['Map Index'] == 5):
        return None  # Neutral map
    else:
        # Determine the Map Picker (team that is NOT the Side Picker)
        if row['Side Picker'] == row['Team 1']:
            return row['Team 2']  # Team 2 is the map picker
        elif row['Side Picker'] == row['Team 2']:
            return row['Team 1']  # Team 1 is the map picker
        else:
            return None  # Default fallback if Side Picker is missing

map_level_data['Map Picker'] = map_level_data.apply(determine_map_picker, axis=1)
map_level_data = map_level_data.drop(columns=['Match Format'])



# Filter match-level data to include only complete matches
match_level_data = match_level_data[match_level_data["Veto Status"] == "Complete"]

# Filter map-level data to only include matches present in the match-level data
map_level_data = map_level_data[map_level_data["Match ID"].isin(match_level_data["Match ID"])]

# Determine halftime scores based on Side Picker and Side Picked
def calculate_halftime_scores(row):
    if row['Side Picker'] == row['Team 1']:
        team1_halftime_score = row['Team 1 Attack Rounds'] if row['Side Picked'] == 'Attack' else row['Team 1 Defense Rounds']
        team2_halftime_score = row['Team 2 Defense Rounds'] if row['Side Picked'] == 'Attack' else row['Team 2 Attack Rounds']
    else:
        team2_halftime_score = row['Team 2 Attack Rounds'] if row['Side Picked'] == 'Attack' else row['Team 2 Defense Rounds']
        team1_halftime_score = row['Team 1 Defense Rounds'] if row['Side Picked'] == 'Attack' else row['Team 1 Attack Rounds']
    return team1_halftime_score, team2_halftime_score
        
map_level_data[['Team 1 Halftime Score', 'Team 2 Halftime Score']] = map_level_data.apply(
    lambda row: pd.Series(calculate_halftime_scores(row)), axis=1
)
map_level_data['Halftime Lead'] = map_level_data['Team 1 Halftime Score'] - map_level_data['Team 2 Halftime Score']



# def validate_starting_sides(row):
#     """
#     Validates the starting sides.
#     Can only be done for games that did not play 24 rounds in regular time.
#     """
#     if row['Team 1 Score'] + row['Team 2 Score'] < 24:
#         # Calculate combinations of rounds to determine pre-swap sides
#         team1_attack = row['Team 1 Attack Rounds']
#         team1_defense = row['Team 1 Defense Rounds']
#         team2_attack = row['Team 2 Attack Rounds']
#         team2_defense = row['Team 2 Defense Rounds']

#         # Check which combination of rounds adds up to 12 (pre-swap sides)
#         if team1_attack + team2_defense == 12:
#             starting_side = "Attack,Defense"
#         elif team1_defense + team2_attack == 12:
#             starting_side = "Defense,Attack"
#         else:
#             return False  # If no valid combination is found, validation fails

#         # Determine which team picked the side based on Side Picker
#         side_picker = row['Side Picker']
#         if side_picker == row['Team 1']:  # Team 1 chose the side
#             side_picked = starting_side.split(",")[0]  # Extract Team 1's side
#         elif side_picker == row['Team 2']:  # Team 2 chose the side
#             side_picked = starting_side.split(",")[1]  # Extract Team 2's side
#         else:
#             return False  # If no valid Side Picker, validation fails

#         if row['Side Picked'] is None:
#             return False  # Cannot validate if Side Picked is missing
#         return side_picked in row['Side Picked']

#     return None

# map_level_data['Starting Side Correct'] = map_level_data.apply(validate_starting_sides, axis=1)

# print(map_level_data[['Match ID', 'Map Name', 'Overtime', 'Starting Side Correct']])



# Save datasets
map_level_data.to_csv("map_level_data_manual.csv", index=False)
match_level_data.to_csv("match_level_data_manual.csv", index=False)

print("Match-Level Data:\n", match_level_data.head())
print("\nMap-Level Data:\n", map_level_data.head())
