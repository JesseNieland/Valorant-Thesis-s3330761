import pandas as pd
from category_encoders import BinaryEncoder

# Load the map-level dataset
map_data = pd.read_csv('map_data_chrono_2.csv')

# One-Hot Encode 'Map Name'
map_data = pd.get_dummies(map_data, columns=['Map Name'], prefix='Map')

# Binary Encode 'Team 1' and 'Team 2'
# Create an encoder instance for binary encoding
team_encoder = BinaryEncoder(cols=['Team 1', 'Team 2'])
map_data = team_encoder.fit_transform(map_data)

# Encode 'Overtime' as binary (1 for True, 0 for False)
map_data['Overtime'] = map_data['Overtime'].astype(int)

# Save the processed dataset
map_data.to_csv("map_data_chrono_2_1.csv", index=False)
