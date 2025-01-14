import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ast

# Load datasets
map_level_data = pd.read_csv('map_level_data_manual copy.csv')
match_level_data = pd.read_csv("match_level_data_manual copy.csv")

# Count of side picks per map
side_picks_count = map_level_data.groupby(["Map Name", "Side Picked"]).size().unstack(fill_value=0)

# Normalize side preferences by map
side_preferences_normalized = side_picks_count.div(side_picks_count.sum(axis=1), axis=0) * 100

# Calculate the overall attack vs defense percentage
overall_attack_percentage = side_preferences_normalized["Attack"].sum() / side_preferences_normalized.values.sum() * 100

# Adjust Score Difference to reflect the side picker's perspective
map_level_data["Adjusted Score Difference"] = map_level_data.apply(
    lambda row: row["Score Difference"] if row["Side Picker"] == row["Team 1"] else -row["Score Difference"],
    axis=1
)

# Group by map and side picked, and calculate the mean adjusted score difference
score_difference_by_side = map_level_data.groupby(["Map Name", "Side Picked"])["Adjusted Score Difference"].mean().unstack(fill_value=0)


# Plot the count of side picks per map
side_picks_count.plot(kind="bar", stacked=False, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title("Count of Side Picks per Map")
plt.ylabel("Count")
plt.xlabel("Map Name")
plt.xticks(rotation=45)
plt.legend(title="Side Picked")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("side_pick_count_per_map.png")
plt.close()

# Plot side preferences
side_preferences_normalized.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Side Preference by Map (Normalized)")
plt.ylabel("Percentage (%)")
plt.xlabel("Map Name")
plt.xticks(rotation=45)
plt.legend(title="Side Picked")
plt.axhline(overall_attack_percentage, color='black', linestyle='--', linewidth=1, label="Overall Balance Line")
plt.text(
    len(side_preferences_normalized) - 1,  # Position near the right edge of the plot
    overall_attack_percentage + 1,  # Slightly above the line
    f"{overall_attack_percentage:.1f}%",
    color="black",
    fontsize=10,
    ha="center",
)
plt.tight_layout()
plt.savefig("side_preferences_by_map_percentage.png")
plt.close()

# Plot average score difference
score_difference_by_side.plot(kind="bar", figsize=(10, 6))
plt.title("Average Score Difference by Side Picked per Map")
plt.ylabel("Average Score Difference")
plt.xlabel("Map Name")
plt.xticks(rotation=45)
plt.legend(title="Side Picked")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig("average_score_difference_by_side_picked.png")
plt.close()


# Calculate the starting side score difference
def calculate_starting_side_difference(row):
    if row["Side Picker"] == row["Team 1"]:
        if row["Side Picked"] == "Attack":
            return row["Team 1 Attack Rounds"] - row["Team 2 Defense Rounds"]
        elif row["Side Picked"] == "Defense":
            return row["Team 1 Defense Rounds"] - row["Team 2 Attack Rounds"]
    elif row["Side Picker"] == row["Team 2"]:
        if row["Side Picked"] == "Attack":
            return row["Team 2 Attack Rounds"] - row["Team 1 Defense Rounds"]
        elif row["Side Picked"] == "Defense":
            return row["Team 2 Defense Rounds"] - row["Team 1 Attack Rounds"]
    return None

map_level_data["Starting Side Score Difference"] = map_level_data.apply(
    calculate_starting_side_difference, axis=1
)

# Group by map and side picked, and calculate the mean starting side score difference
starting_side_difference_by_map = map_level_data.groupby(
    ["Map Name", "Side Picked"]
)["Starting Side Score Difference"].mean().unstack(fill_value=0)

# Plot starting side score difference
starting_side_difference_by_map.plot(kind="bar", figsize=(10, 6))
plt.title("Average Starting Side Score Difference by Side Picked per Map")
plt.ylabel("Average Score Difference")
plt.xlabel("Map Name")
plt.xticks(rotation=45)
plt.legend(title="Side Picked")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig("average_starting_side_score_difference_per_map.png")
plt.close()

# Calculate the total rounds won on attack and defense for each map
side_score_difference = map_level_data.groupby("Map Name")[[
    "Team 1 Attack Rounds", "Team 2 Attack Rounds", 
    "Team 1 Defense Rounds", "Team 2 Defense Rounds"
]].sum()

# Compute total attack and defense rounds for each map
side_score_difference["Attack Rounds Won"] = (
    side_score_difference["Team 1 Attack Rounds"] + side_score_difference["Team 2 Attack Rounds"]
)
side_score_difference["Defense Rounds Won"] = (
    side_score_difference["Team 1 Defense Rounds"] + side_score_difference["Team 2 Defense Rounds"]
)

# Add a column for the total maps played per map
maps_played = map_level_data["Map Name"].value_counts()

# Normalize the score difference by the total maps played
side_score_difference["Average Side Score Difference"] = (
    (side_score_difference["Attack Rounds Won"] - side_score_difference["Defense Rounds Won"])
    / maps_played
)

# Plot the average score difference
plt.figure(figsize=(10, 6))
side_score_difference["Average Side Score Difference"].plot(kind="bar", color="skyblue")
plt.title("Average Score Difference Between Sides per Map")
plt.ylabel("Average Score Difference (Attack - Defense)")
plt.xlabel("Map Name")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("average_score_difference_between_sides.png")
plt.close()

# Sum up the required columns for attack and defense rounds by map
side_rounds = map_level_data.groupby("Map Name")[
    ["Team 1 Attack Rounds", "Team 1 Defense Rounds", "Team 2 Attack Rounds", "Team 2 Defense Rounds"]
].sum()

# Calculate total rounds played in regular time (so excluding overtime)
side_rounds["Total Rounds"] = (
    side_rounds["Team 1 Attack Rounds"]
    + side_rounds["Team 1 Defense Rounds"]
    + side_rounds["Team 2 Attack Rounds"]
    + side_rounds["Team 2 Defense Rounds"]
)

# Calculate attack and defense winrates
side_rounds["Attack Winrate"] = (
    (side_rounds["Team 1 Attack Rounds"] + side_rounds["Team 2 Attack Rounds"])
    / side_rounds["Total Rounds"]
)
side_rounds["Defense Winrate"] = (
    (side_rounds["Team 1 Defense Rounds"] + side_rounds["Team 2 Defense Rounds"])
    / side_rounds["Total Rounds"]
)

winrate_by_side = side_rounds[["Attack Winrate", "Defense Winrate"]]

# Plot Average Winrate by Side
winrate_by_side.plot(kind="bar", figsize=(12, 6))
plt.title("Average Winrate on Attack and Defense per Map (Excluding Overtime)")
plt.ylabel("Winrate (%)")
plt.xlabel("Map Name")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(title="Side")
plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="50% Winrate")
plt.tight_layout()
plt.savefig("average_winrate_by_side_per_map.png")
plt.close()

# Calculate total count of maps played
map_play_count = map_level_data["Map Name"].value_counts()

# Plot Total Count of Maps Played
map_play_count.plot(kind="bar", figsize=(12, 6), color="skyblue")
plt.title("Total Count of Maps Played")
plt.ylabel("Count")
plt.xlabel("Map Name")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("total_count_of_maps_played.png")
plt.close()

# Define the custom order for the game categories
category_order = ["Overtime", "Closely Contested", "Competitive", "Blowouts"]

# Exclude overtime games when calculating thresholds
non_overtime_games = map_level_data[map_level_data["Overtime"] == 0]

# Calculate percentiles for the absolute Score Difference from non-overtime games
low_threshold = non_overtime_games["Score Difference"].abs().quantile(0.25)
high_threshold = non_overtime_games["Score Difference"].abs().quantile(0.75)

# Categorize games based on percentiles (excluding overtime for threshold calculation)
def categorize_games_percentiles_no_overtime(row):
    if row["Overtime"] > 0:
        return "Overtime"
    elif abs(row["Score Difference"]) <= low_threshold:
        return "Closely Contested"
    elif abs(row["Score Difference"]) >= high_threshold:
        return "Blowouts"
    else:
        return "Competitive"

# Apply categorization to the dataset
map_level_data["Game Category"] = map_level_data.apply(categorize_games_percentiles_no_overtime, axis=1)

# Ensure the Game Category column has the custom order
map_level_data["Game Category"] = pd.Categorical(
    map_level_data["Game Category"],
    categories=category_order,
    ordered=True
)

# Calculate percentages for each category per map
category_percentage = (
    map_level_data.groupby(["Map Name", "Game Category"], observed=True)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=category_order)
    .apply(lambda x: x / x.sum() * 100, axis=1)
)

# Plotting the stacked bar chart
category_percentage.plot(kind="bar", stacked=True, figsize=(12, 8))
plt.title("Game Distribution by Map (Overtime, Closely Contested, Blowouts, Competitive)")
plt.ylabel("Percentage of Games")
plt.xlabel("Map Name")
plt.xticks(rotation=45)
plt.legend(title="Game Category", loc="upper right")
plt.tight_layout()
plt.savefig("game_distribution_by_map.png")
plt.close()

# Function to parse the Map Veto column
def parse_map_veto(veto_list):
    # Convert string to list of dictionaries
    try:
        veto_data = ast.literal_eval(veto_list)
        return veto_data
    except (ValueError, SyntaxError):
        return []

# Apply parsing to the Map Veto column
match_level_data["Parsed Veto"] = match_level_data["Map Veto"].apply(parse_map_veto)

# Flatten the parsed veto data
veto_actions = []
for step in match_level_data["Parsed Veto"]:
    for action in step:
        veto_actions.append(action)

# Create a DataFrame from the veto actions
veto_df = pd.DataFrame(veto_actions)

# Calculate percentages for each map and action
veto_summary = (
    veto_df.groupby(["Map", "Action"])
    .size()
    .unstack(fill_value=0)
    .apply(lambda x: x / x.sum() * 100, axis=1)  # Normalize to percentages
)

# Plot the frequency of actions per map
veto_summary.plot(kind="bar", figsize=(12, 6))
plt.title("Map Action Frequency (Normalized to Percentage)")
plt.xlabel("Map")
plt.ylabel("Percentage")
plt.legend(title="Action", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("map_action_frequency.png")
plt.close()

# Merge map-level data with match-level data to include the Patch column
merged_data = map_level_data.merge(match_level_data[['Match ID', 'Patch']], on='Match ID')

# Ensure patches are sorted properly
merged_data['Patch'] = pd.Categorical(merged_data['Patch'], categories=sorted(merged_data['Patch'].unique()), ordered=True)

# Group data by Map Name, Patch, and Side Picked to count side picks
side_picks_patch_count = merged_data.groupby(["Map Name", "Patch", "Side Picked"], observed=False).size().unstack(fill_value=0)

# Normalize side preferences by map and patch
side_preferences_patch_normalized = side_picks_patch_count.div(side_picks_patch_count.sum(axis=1), axis=0) * 100

# Ensure all patches are represented for each map
all_patches = sorted(merged_data['Patch'].unique())
all_maps = sorted(merged_data['Map Name'].unique())
side_preferences_patch_normalized = side_preferences_patch_normalized.reindex(
    pd.MultiIndex.from_product([all_maps, all_patches], names=["Map Name", "Patch"]),
    fill_value=0
)

# Create subplots
num_maps = len(all_maps)
fig, axes = plt.subplots(nrows=num_maps, ncols=1, figsize=(10, num_maps * 3), sharex=True)

# Loop through each map and create a subplot
for ax, map_name in zip(axes, all_maps):
    map_data = side_preferences_patch_normalized.loc[map_name].reset_index()
    map_data['Patch'] = pd.Categorical(map_data['Patch'], categories=all_patches, ordered=True)
    
    # Plot data for the current map
    map_data.plot(kind="line", marker='o', ax=ax)
    
    # Customize each subplot
    ax.set_title(map_name)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ax.set_xticks(range(len(all_patches)))
    ax.set_xticklabels(all_patches)

# Global x-axis label
plt.xlabel("Patch")
plt.tight_layout()
plt.savefig("side_preference_progression_subplots.png")
plt.close()



# Merge map-level data with match-level data to include the Patch column
merged_data = map_level_data.merge(match_level_data[['Match ID', 'Patch']], on='Match ID')

# Ensure patches are sorted properly
merged_data['Patch'] = pd.Categorical(merged_data['Patch'], categories=sorted(merged_data['Patch'].unique()), ordered=True)

# Count maps played per patch
maps_per_patch = merged_data.groupby('Patch', observed=True).size()

# Plot the count of maps played per patch
maps_per_patch.plot(kind='bar', color='skyblue', figsize=(10, 6))
plt.title("Count of Maps Played Per Patch")
plt.ylabel("Count of Maps")
plt.xlabel("Patch")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("maps_count_per_patch.png")
plt.close()



# Count maps played per map and patch
maps_per_map_patch = merged_data.groupby(['Map Name', 'Patch'], observed=False).size().unstack(fill_value=0)

# Determine the number of maps
num_maps = len(maps_per_map_patch.index)

# Create subplots for each map
fig, axes = plt.subplots(nrows=num_maps, ncols=1, figsize=(10, num_maps * 3), sharex=True)

# Loop through maps and create a subplot for each
for ax, map_name in zip(axes, maps_per_map_patch.index):
    maps_per_map_patch.loc[map_name].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(map_name)
    ax.set_ylabel("Count of Maps")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a shared x-axis label
plt.xlabel("Patch")
plt.tight_layout()
plt.savefig("maps_count_per_patch_subplots.png")
plt.close()


# Calculate average side score difference (Attack - Defense) per map and patch
merged_data['Side Difference'] = (
    (merged_data['Team 1 Attack Rounds'] + merged_data['Team 2 Attack Rounds']) -
    (merged_data['Team 1 Defense Rounds'] + merged_data['Team 2 Defense Rounds'])
)

avg_side_diff_per_map_patch = (
    merged_data.groupby(['Map Name', 'Patch'], observed=False)['Side Difference']
    .mean()
    .unstack(fill_value=0)
)

# Create subplots for each map
fig, axes = plt.subplots(nrows=num_maps, ncols=1, figsize=(10, num_maps * 3), sharex=True)

# Plot for each map in its respective subplot
for ax, map_name in zip(axes, avg_side_diff_per_map_patch.index):
    avg_side_diff_per_map_patch.loc[map_name].plot(kind='line', marker='o', ax=ax)
    
    # Customize each subplot
    ax.set_title(f"{map_name}")
    ax.set_ylabel("Avg Side Diff (Atk - Def)")
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Global x-axis label and adjustments
plt.xlabel("Patch")
plt.xticks(ticks=range(len(avg_side_diff_per_map_patch.columns)), labels=avg_side_diff_per_map_patch.columns, rotation=45)
plt.tight_layout()

# Save the compact plot
plt.savefig("avg_side_diff_compact_subplots.png")
plt.close()


# Calculate total rounds and wins for attack and defense
merged_data['Total Rounds'] = merged_data['Team 1 Attack Rounds'] + merged_data['Team 1 Defense Rounds'] + \
                              merged_data['Team 2 Attack Rounds'] + merged_data['Team 2 Defense Rounds']
merged_data['Attack Rounds Won'] = merged_data['Team 1 Attack Rounds'] + merged_data['Team 2 Attack Rounds']
merged_data['Defense Rounds Won'] = merged_data['Team 1 Defense Rounds'] + merged_data['Team 2 Defense Rounds']

# Group by Map and Patch to compute win rates
win_rates = merged_data.groupby(['Map Name', 'Patch'], observed=False).agg({
    'Attack Rounds Won': 'sum',
    'Defense Rounds Won': 'sum',
    'Total Rounds': 'sum'
})
win_rates['Attack Win Rate (%)'] = (win_rates['Attack Rounds Won'] / win_rates['Total Rounds']) * 100
win_rates['Defense Win Rate (%)'] = (win_rates['Defense Rounds Won'] / win_rates['Total Rounds']) * 100

# Pivot data for heatmaps
attack_win_rates = win_rates['Attack Win Rate (%)'].unstack(fill_value=0)
defense_win_rates = win_rates['Defense Win Rate (%)'].unstack(fill_value=0)

# Create heatmaps for attack and defense win rates
for side, data in {'Attack': attack_win_rates, 'Defense': defense_win_rates}.items():
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Win Rate (%)'}, linewidths=0.5)
    
    # Customizing the plot
    plt.title(f"{side} Win Rates Across Maps and Patches")
    plt.ylabel("Map Name")
    plt.xlabel("Patch")
    plt.tight_layout()
    plt.savefig(f"{side.lower()}_win_rates_heatmap.png")
    plt.close()
