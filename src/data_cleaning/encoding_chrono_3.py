import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder

# Load dataset
map_data = pd.read_csv("map_data_chrono_2.csv")

# One-Hot Encode Map Name
# Store map names to keep them in the final dataset
map_names = map_data['Map Name']
map_data.rename(columns={'Map Name': 'Map'}, inplace=True)

# One-hot encode 'Map Name' and convert boolean to integer (0/1)
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
map_name_encoded = one_hot_encoder.fit_transform(map_data[['Map']])
map_name_df = pd.DataFrame(map_name_encoded, columns=one_hot_encoder.get_feature_names_out(['Map'])).astype(int)

# Binary Encode Teams
# Store team names to retain in final dataset
teams = map_data[['Team 1', 'Team 2']]

# Apply binary encoding to 'Team 1' and 'Team 2'
binary_encoder = BinaryEncoder()
team_1_encoded = binary_encoder.fit_transform(map_data['Team 1'])
team_2_encoded = binary_encoder.fit_transform(map_data['Team 2'])

# Adjust column names to avoid redundancy and indicate encoding type
team_1_encoded.columns = [f"{col}" for col in team_1_encoded.columns]
team_2_encoded.columns = [f"{col}" for col in team_2_encoded.columns]

# Combine Encoded Columns with Original Data
# Drop original 'Map Name', 'Team 1', and 'Team 2' columns before concatenating
map_data = map_data.drop(columns=['Map', 'Team 1', 'Team 2'])

# Concatenate the original columns, one-hot encoded maps, binary encoded teams, and any remaining columns
final_data = pd.concat([map_data, map_name_df, team_1_encoded, team_2_encoded, map_names, teams], axis=1)

# Ensure 'Overtime' is in 0/1 integer format
final_data['Overtime'] = final_data['Overtime'].astype(int)

final_data.to_csv("map_data_chrono_2_2.csv", index=False)
