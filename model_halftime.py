import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

map_level_data = pd.read_csv("map_level_data_prepared.csv")
match_level_data = pd.read_csv("match_level_data_manual copy.csv")

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the Halftime Lead feature
map_level_data['Halftime Lead Scaled'] = scaler.fit_transform(
    map_level_data[['Halftime Lead']]
)

# One-hot encode the map names
map_dummies = pd.get_dummies(map_level_data['Map Name'], prefix='Map')

for col in map_dummies.columns:
    map_level_data[f'Pick_{col}'] = map_dummies[col]


# CODE FOR OLD FEATURE
# # Initialize Team 1 and Team 2 map pick columns with 0 (False)
# map_level_data['Attack_MapPick'] = 0
# map_level_data['Defense_MapPick'] = 0

# # Set Team 1 map pick to 1 where Team 1 picked the map
# map_level_data.loc[map_level_data['Side Picked'] == 'Attack', 'Attack_MapPick'] = 1

# # Set Team 2 map pick to 1 where Team 2 picked the map
# map_level_data.loc[map_level_data['Side Picked'] == 'Defense', 'Defense_MapPick'] = 1

# # Combine Team 1 and Team 2 map picks with map names
# for col in map_dummies.columns:
#     map_level_data[f'Attack_{col}'] = map_level_data['Attack_MapPick'] * map_dummies[col]
#     map_level_data[f'Defense_{col}'] = map_level_data['Defense_MapPick'] * map_dummies[col]

# # Drop intermediate columns
# map_level_data.drop(['Attack_MapPick', 'Defense_MapPick'], axis=1, inplace=True)


# Filter tied games
map_level_data_filtered = map_level_data[map_level_data['Halftime Lead'] == 0].copy()
# map_level_data_filtered = map_level_data.copy()

# Prepare features and target for the decision tree
map_level_data_filtered['Target'] = map_level_data_filtered.apply(
    lambda row: 0 if row["Map Winner"] == row["Team X"] else 1,
    axis=1
)
map_level_data_filtered['Picked_Attack'] = map_level_data_filtered.apply(
    lambda row: 1 if row["Side Picked"] == "Attack" else 0,
    axis=1
)
map_level_data_filtered["Neutral Map"] = map_level_data_filtered.apply(
    lambda row: 1 if pd.isna(row["Map Picker"]) else 0, axis=1
)


new_rare_maps = ["Abyss", "Breeze", "Fracture", "Pearl", "Sunset"]
map_level_data_filtered["New/Rare"] = map_level_data_filtered.apply(
    lambda row: 1 if row["Map Name"] in new_rare_maps else 0, axis=1
)

attack_maps = ["Icebox", "Lotus"]
defense_maps = ["Abyss", "Ascent"]
map_level_data_filtered["Side Strength"] = map_level_data_filtered.apply(
    lambda row: 1 if row["Map Name"] in attack_maps 
    else -1 if row["Map Name"] in defense_maps
    else 0, axis=1
)

# Select features for modeling
# Define the initial set of features for the simple model
initial_features = ['Halftime Lead Scaled', 'Halftime Lead', 'Neutral Map', 'Picked_Attack', 'New/Rare', 'Side Strength']

# Include encoded categorical variables for Map Picker and Name
# initial_features.extend([col for col in map_level_data.columns if "Attack_Map_" in col or "Defense_Map_" in col])
initial_features.extend([col for col in map_level_data.columns if "Pick_Map_" in col])

# Extract the feature matrix (X) and the target variable (y)
X = map_level_data_filtered[initial_features]
y = map_level_data_filtered['Target']

# Split into train and test set including the Halftime Lead feature (Halftime Lead Scaled is dropped)
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Halftime Lead Scaled']), y, test_size=0.2, random_state=42)

# Train the decision tree
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)

# Predict on both training and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the results
print("Decision Tree - Training Accuracy:", f"{train_accuracy * 100:.2f}%")
print("Decision Tree - Test Accuracy:", f"{test_accuracy * 100:.2f}%")
print("Classification Report Model:\n", classification_report(y_test, y_test_pred))

# Display confusion matrix
print("Confusion Matrix Model:")
print(confusion_matrix(y_test, y_test_pred))

# Apply heuristic on the test set
def halftime_heuristic(row):
    if row['Halftime Lead'] > 0:
        return 0  # Team X
    elif row['Halftime Lead'] < 0:
        return 1  # Team Y
    else:
        return 1 if row['Neutral Map'] == 1 else 0  # Team X if not neutral, Team Y if neutral

# Predict using the heuristic
y_heuristic = X_test.apply(halftime_heuristic, axis=1)
y_heuristic_train = X_train.apply(halftime_heuristic, axis=1)

# Evaluate the heuristic
print("Heuristic Accuracy on Train Set:", accuracy_score(y_train, y_heuristic_train))
print("Heuristic Accuracy on Test Set:", accuracy_score(y_test, y_heuristic))
print("Classification Report Heuristic:\n", classification_report(y_test, y_heuristic, zero_division=0))

# Display confusion matrix
print("Confusion Matrix Heuristic:")
print(confusion_matrix(y_test, y_heuristic))

# Visualize the tree
plt.figure(figsize=(12, 8), dpi=300)
plot_tree(model, feature_names=X.drop(columns=['Halftime Lead Scaled']).columns, class_names=['Team X', 'Team Y'], filled=True)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# Track accuracy for different max depths
depths = range(1, 16)  # Depths to test
train_accuracies = []
test_accuracies = []

# Loop through depths
for depth in depths:
    # Train the decision tree model
    loop_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    loop_model.fit(X_train, y_train)
    
    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, loop_model.predict(X_train))
    test_accuracy = accuracy_score(y_test, loop_model.predict(X_test))
    
    # Store the accuracies
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(depths, test_accuracies, label='Test Accuracy', marker='o')
plt.title('Effect of max_depth on Decision Tree Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0.7, 0.9)
plt.savefig('decision_tree_accuracy_vs_depth.png', bbox_inches='tight')
plt.close()



# Initialize Random Forest with some hyperparameters
rf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Display feature importances
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances)

# Make predictions
rf_train_preds = rf_model.predict(X_train)
rf_test_preds = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest - Training Accuracy:", accuracy_score(y_train, rf_train_preds))
print("Random Forest - Test Accuracy:", accuracy_score(y_test, rf_test_preds))
print("Classification Report (Test):\n", classification_report(y_test, rf_test_preds))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, rf_test_preds))

# Track accuracy for different max depths
depths_rf = range(1, 16)  # Depths to test
train_accuracies_rf = []
test_accuracies_rf = []

# Loop through depths
for depth in depths_rf:
    # Train the decision tree model
    loop_rf_model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    loop_rf_model.fit(X_train, y_train)
    
    # Calculate training and test accuracy
    train_accuracy_rf = accuracy_score(y_train, loop_rf_model.predict(X_train))
    test_accuracy_rf = accuracy_score(y_test, loop_rf_model.predict(X_test))
    
    # Store the accuracies
    train_accuracies_rf.append(train_accuracy_rf)
    test_accuracies_rf.append(test_accuracy_rf)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies_rf, label='Training Accuracy (RF)', marker='o')
plt.plot(depths, test_accuracies_rf, label='Test Accuracy (RF)', marker='o')
plt.title('Effect of max_depth on Random Forest Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0.7, 0.9)
plt.savefig('random_forest_accuracy_vs_depth.png', bbox_inches='tight')
plt.close()

# Split into train and test set including the Halftime Lead Scaled feature (Halftime Lead is dropped)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X.drop(columns=['Halftime Lead']), y, test_size=0.2, random_state=42)

# Initialize SVM with some hyperparameters
svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)

# Train the model
svm_model.fit(X_train_svm, y_train_svm)

# Make predictions
svm_train_preds = svm_model.predict(X_train_svm)
svm_test_preds = svm_model.predict(X_test_svm)

# Evaluate the model
print("SVM - Training Accuracy:", accuracy_score(y_train_svm, svm_train_preds))
print("SVM - Test Accuracy:", accuracy_score(y_test_svm, svm_test_preds))
print("Classification Report (Test):\n", classification_report(y_test_svm, svm_test_preds))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test_svm, svm_test_preds))



# Parameters
seeds = range(10)  # Random seeds for data splits
depths = range(1, 16)  # Depths to test

# Initialize arrays to store accuracies
train_accuracies_rf = np.zeros((len(seeds), len(depths)))
test_accuracies_rf = np.zeros((len(seeds), len(depths)))
train_accuracies_dt = np.zeros((len(seeds), len(depths)))
test_accuracies_dt = np.zeros((len(seeds), len(depths)))
heuristic_accuracies = []

# Initialize a dictionary to store accuracies for each kernel
kernels = ['linear', 'rbf', 'poly', 'sigmoid']  # Kernels to compare
results = {kernel: {'train': [], 'test': []} for kernel in kernels}

# Loop over random seeds for data splits
for seed_idx, seed in enumerate(seeds):
    # Split the dataset with the current random seed
    # Split into train and test set including the Halftime Lead feature (Halftime Lead Scaled is dropped)
    X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Halftime Lead Scaled']), y, test_size=0.2, random_state=seed)
    # Split into train and test set including the Halftime Lead Scaled feature (Halftime Lead is dropped)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X.drop(columns=['Halftime Lead']), y, test_size=0.2, random_state=seed)

    y_heuristic = X_test.apply(halftime_heuristic, axis=1)
    
    # Record accuracies
    heuristic_accuracies.append(accuracy_score(y_test, y_heuristic))
    
    for depth_idx, depth in enumerate(depths):
        # Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)  # Keep RF seed fixed
        rf_model.fit(X_train, y_train)
        train_accuracies_rf[seed_idx, depth_idx] = accuracy_score(y_train, rf_model.predict(X_train))
        test_accuracies_rf[seed_idx, depth_idx] = accuracy_score(y_test, rf_model.predict(X_test))
        
        # Decision Tree Model
        dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)  # Keep DT seed fixed
        dt_model.fit(X_train, y_train)
        train_accuracies_dt[seed_idx, depth_idx] = accuracy_score(y_train, dt_model.predict(X_train))
        test_accuracies_dt[seed_idx, depth_idx] = accuracy_score(y_test, dt_model.predict(X_test))
    
    for kernel in kernels:
        # Train SVM with the current kernel
        svm_model = SVC(kernel=kernel, C=1, random_state=42)  # Adjust hyperparameters as needed
        svm_model.fit(X_train_svm, y_train_svm)
        
        # Record accuracies for the current kernel and seed
        train_accuracy = accuracy_score(y_train_svm, svm_model.predict(X_train_svm))
        test_accuracy = accuracy_score(y_test_svm, svm_model.predict(X_test_svm))
        results[kernel]['train'].append(train_accuracy)
        results[kernel]['test'].append(test_accuracy)

# Calculate mean and standard deviation of accuracies for Random Forest
mean_train_accuracies_rf = train_accuracies_rf.mean(axis=0)
std_train_accuracies_rf = train_accuracies_rf.std(axis=0)
mean_test_accuracies_rf = test_accuracies_rf.mean(axis=0)
std_test_accuracies_rf = test_accuracies_rf.std(axis=0)

# Calculate mean and standard deviation of accuracies for Decision Tree
mean_train_accuracies_dt = train_accuracies_dt.mean(axis=0)
std_train_accuracies_dt = train_accuracies_dt.std(axis=0)
mean_test_accuracies_dt = test_accuracies_dt.mean(axis=0)
std_test_accuracies_dt = test_accuracies_dt.std(axis=0)

# Plot results for Random Forest
plt.figure(figsize=(12, 6))
plt.errorbar(depths, mean_train_accuracies_rf, yerr=std_train_accuracies_rf, label='Training Accuracy (RF)', fmt='-o', capsize=5)
plt.errorbar(depths, mean_test_accuracies_rf, yerr=std_test_accuracies_rf, label='Test Accuracy (RF)', fmt='-o', capsize=5)
plt.title('Effect of max_depth on Random Forest Accuracy Across Data Splits')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0.7, 0.9)
plt.savefig('random_forest_accuracy_vs_depth_splits.png', bbox_inches='tight')
plt.close()

# Plot results for Decision Tree
plt.figure(figsize=(12, 6))
plt.errorbar(depths, mean_train_accuracies_dt, yerr=std_train_accuracies_dt, label='Training Accuracy (DT)', fmt='-o', capsize=5)
plt.errorbar(depths, mean_test_accuracies_dt, yerr=std_test_accuracies_dt, label='Test Accuracy (DT)', fmt='-o', capsize=5)
plt.title('Effect of max_depth on Decision Tree Accuracy Across Data Splits')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0.7, 0.9)
plt.savefig('decision_tree_accuracy_vs_depth_splits.png', bbox_inches='tight')
plt.close()

# Compare best depths
best_depth_rf = depths[np.argmax(mean_test_accuracies_rf)]
best_depth_rf_accuracy = mean_test_accuracies_rf[np.argmax(mean_test_accuracies_rf)] * 100
best_depth_rf_std = std_test_accuracies_rf[np.argmax(mean_test_accuracies_rf)] * 100
best_depth_dt = depths[np.argmax(mean_test_accuracies_dt)]
best_depth_dt_accuracy = mean_test_accuracies_dt[np.argmax(mean_test_accuracies_dt)] * 100
best_depth_dt_std = std_test_accuracies_dt[np.argmax(mean_test_accuracies_dt)] * 100 

mean_heuristic_accuracy = np.mean(heuristic_accuracies) * 100
std_heuristic_accuracy = np.std(heuristic_accuracies) * 100

# Print results
print(f"Best max_depth for Random Forest: {best_depth_rf} with mean test accuracy: {best_depth_rf_accuracy:.2f}% ± {best_depth_rf_std:.2f}%")
print(f"Best max_depth for Decision Tree: {best_depth_dt} with mean test accuracy: {best_depth_dt_accuracy:.2f}% ± {best_depth_dt_std:.2f}%")
print(f"Heuristic Mean Test Accuracy: {mean_heuristic_accuracy:.2f}% ± {std_heuristic_accuracy:.2f}%")

# Calculate mean and standard deviation for each kernel
for kernel in kernels:
    mean_test_accuracy = np.mean(results[kernel]['test']) * 100
    std_test_accuracy = np.std(results[kernel]['test']) * 100
    
    # Print results for this kernel
    print(f"Kernel: {kernel}")
    print(f"  Mean Test Accuracy: {mean_test_accuracy:.2f}% ± {std_test_accuracy:.2f}%")
