import random
import pandas as pd
from sklearn.metrics import accuracy_score

random.seed(42)

df = pd.read_csv("map_level_data_prepared.csv")
match_level_data = pd.read_csv("match_level_data_manual copy.csv")

# Predict the leading team wins if the halftime lead > 0
def predict_winner_no_ties(row):
    if row['Halftime Lead'] > 0:
        return row['Team X']
    elif row['Halftime Lead'] < 0:
        return row['Team Y']
    else:
        return 'Uncertain'  # Tied halftime scores

df['Heuristic Prediction (No Ties)'] = df.apply(predict_winner_no_ties, axis=1)

# Filter out 'Uncertain' predictions for accuracy calculation
df_filtered = df[df['Heuristic Prediction (No Ties)'] != 'Uncertain']

# Calculate accuracy
accuracy = accuracy_score(df_filtered['Map Winner'], df_filtered['Heuristic Prediction (No Ties)'])
print(f"Accuracy of the halftime score differential heuristic excluding ties: {accuracy:.2%} " 
      f"({len(df_filtered)} games)")

def predict_winner(row):
    if row['Halftime Lead'] > 0:
        return row['Team X']
    elif row['Halftime Lead'] < 0:
        return row['Team Y']
    else:  # For tied halftime scores
        # return row['Team X']
        if pd.notnull(row['Map Picker']):  # Use Map Picker if available
            return row['Map Picker']
        elif pd.notnull(row['Side Picker']):  # Use Side Picker if no Map Picker
            return row['Side Picker']
        else:
            return 'Uncertain'  # Fallback if all else fails

# Apply the heuristic to the DataFrame
df['Heuristic Prediction'] = df.apply(predict_winner, axis=1)

# Calculate accuracy
accuracy = accuracy_score(df['Map Winner'], df['Heuristic Prediction'])
print(f"Accuracy of the halftime score differential heuristic: {accuracy:.2%} "
      f"({len(df)} games)")


# Filter tied games
tied_games = df[df['Halftime Lead'] == 0]

# Accuracy when predicting team X
map_picker_predictions = tied_games['Team X']
map_picker_accuracy = accuracy_score(tied_games['Map Winner'], map_picker_predictions)

print(f"Team X Predictor Accuracy (tied halftime): {map_picker_accuracy:.2%} "
      f"({len(tied_games)} games)")

# Map Picker Accuracy when Map Picker exists
map_picker_games = tied_games[tied_games['Map Picker'].notnull()]
map_picker_predictions = map_picker_games['Map Picker']
map_picker_accuracy = accuracy_score(map_picker_games['Map Winner'], map_picker_predictions)

print(f"Map Picker Accuracy (tied halftime): {map_picker_accuracy:.2%} "
      f"({len(map_picker_games)} games)")

# Side Picker Accuracy when there is no Map Picker
side_picker_games = tied_games[tied_games['Map Picker'].isnull()]
side_picker_predictions = side_picker_games['Side Picker']

side_picker_accuracy = accuracy_score(side_picker_games['Map Winner'], side_picker_predictions)

print(f"Side Picker Accuracy (neutral map, tied halftime): {side_picker_accuracy:.2%} "
      f"({len(side_picker_games)} games)")


# Heuristic function that predicts randomly for tied halftime scores
def predict_winner_random(row):
    if row['Halftime Lead'] > 0:
        return row['Team X']
    elif row['Halftime Lead'] < 0:
        return row['Team Y']
    else:  # For tied halftime scores
        return random.choice([row['Team X'], row['Team Y']])

df['Heuristic Prediction (random)'] = df.apply(predict_winner_random, axis=1)

heuristic_accuracy = accuracy_score(df['Map Winner'], df['Heuristic Prediction (random)'])
print(f"Heuristic Accuracy (random): {heuristic_accuracy:.2%}")
