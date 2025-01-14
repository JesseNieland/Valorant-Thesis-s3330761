import pandas as pd

# Load the datasets
match_summary = pd.read_csv('match_summary_dataset_with_winner.csv')
map_data = pd.read_csv('map_level_dataset.csv')

# Ensure date columns are in datetime format
match_summary['Date'] = pd.to_datetime(match_summary['Date'])
map_data['Date'] = pd.to_datetime(map_data['Date'], errors='coerce')  # Convert, ignoring errors if any

# Get the earliest date for each event to determine chronological event order
event_start_dates = match_summary.groupby('Event')['Date'].min().reset_index()
event_start_dates = event_start_dates.sort_values(by='Date', ascending=True).reset_index(drop=True)

# Add a chronological rank to each event based on the start date
event_start_dates['Event_Rank'] = range(1, len(event_start_dates) + 1)

# Merge this rank back into match_summary
match_summary = match_summary.merge(event_start_dates[['Event', 'Event_Rank']], on='Event', how='left')

# Sort match_summary by event rank and within that by date
match_summary = match_summary.sort_values(by=['Event_Rank', 'Date']).reset_index(drop=True)

# Assign new sequential Match IDs in the order of the rows in match_summary
match_summary['New_Match_ID'] = range(1, len(match_summary) + 1)

# Create a dictionary to map old Match IDs to the new Match IDs
id_mapping = dict(zip(match_summary['Match ID'], match_summary['New_Match_ID']))

# Map the old Match IDs in map_data to the new Match IDs
map_data['Match ID'] = map_data['Match ID'].map(id_mapping)

# Make 'Match ID' the first column in map_data
map_data = map_data[['Match ID'] + [col for col in map_data.columns if col != 'Match ID']]

# Sort map_data by Match ID only, preserving row order within each Match ID group
map_data = map_data.sort_values(by='Match ID', kind='stable').reset_index(drop=True)

# Drop temporary columns in match_summary and rename 'New_Match_ID' to 'Match ID'
match_summary = match_summary.drop(columns=['Event_Rank', 'Match ID']).rename(columns={'New_Match_ID': 'Match ID'})
match_summary = match_summary[['Match ID'] + [col for col in match_summary.columns if col != 'Match ID']]

match_summary.to_csv('match_summary_chronological.csv', index=False)
map_data.to_csv('map_data_chronological.csv', index=False)
