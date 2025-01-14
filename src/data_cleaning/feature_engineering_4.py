import pandas as pd

# Load dataset
data = pd.read_csv('map_data_chrono_3_3_1.csv')

# Team Encoding
# Create a mapping of team names to unique integers
team_mapping = {team: idx for idx, team in enumerate(pd.concat([data['Team 1'], data['Team 2']]).unique(), start=1)}

# Create new columns for team IDs
data['Team 1 ID'] = data['Team 1'].map(team_mapping)
data['Team 2 ID'] = data['Team 2'].map(team_mapping)

# Remove Binary Encoded Team Columns
# Identify and drop columns with binary-encoded teams (Team 1_0, Team 1_1, etc.)
team_columns = [col for col in data.columns if col.startswith('Team 1_') or col.startswith('Team 2_')]
data = data.drop(columns=team_columns)

# Adjust Score Difference for Overtime
# Set score difference to 0 for overtime matches
data.loc[data['Overtime'] == 1, 'Score Difference'] = 0

# Combine Attack and Defense Round Differences
# Add a new column for total round difference
data['Total Rounds Played Difference'] = data['Attack Rounds Played Difference'] + data['Defense Rounds Played Difference']

# Drop the redundant columns
data = data.drop(columns=['Attack Rounds Played Difference', 'Defense Rounds Played Difference', 'Overtime'])

data.to_csv('map_data_chrono_3_3_2.csv', index=False)

# Save the team mapping to a separate file
team_mapping_df = pd.DataFrame(list(team_mapping.items()), columns=['Team Name', 'Team ID'])
team_mapping_df.to_csv('team_mapping.csv', index=False)

print(data.head())
print("Team mapping saved as 'team_mapping.csv'")
