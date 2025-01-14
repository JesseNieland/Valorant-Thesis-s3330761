import pandas as pd

# Load datasets
match_summary = pd.read_csv('match_summary_chronological.csv')
map_data = pd.read_csv('map_data_chronological.csv')

map_data['Overtime'] = ((map_data['Team 1 Score'] >= 14) | (map_data['Team 2 Score'] >= 14))

# Calculate score difference as the target variable and drop the individual scores
map_data['Score Difference'] = map_data['Team 1 Score'] - map_data['Team 2 Score']
# map_data = map_data.drop(columns=['Team 1 Score', 'Team 2 Score'])

map_data = map_data.drop(columns=['Map Duration', 'Team 1 Score', "Team 2 Score"])

# Save the processed data to a new CSV file
map_data.to_csv("map_data_chrono_2.csv", index=False)
