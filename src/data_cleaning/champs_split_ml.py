import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('data/map_data_chrono_3_3_1.csv')

# Separate features and target
X = data.drop(columns=['Score Difference', 'Team 1', 'Team 2', 'Map Name'], axis=1)
y = data['Score Difference']

# Split the data using time-based split, testing on the Champions event
train_size = 1020  # Champions starts from row 1020
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')

# Create a DataFrame with the test set results, including team names and map names
result_df = X_test.copy()  # Copy the test data
result_df = pd.DataFrame(result_df, columns=X.columns)  # Ensure column names are retained

# Add the predicted score difference, actual score difference, team names, and map names to the results
result_df['Predicted Score Difference'] = y_pred
result_df['Actual Score Difference'] = y_test
result_df['Team 1'] = data.loc[train_size:, 'Team 1'].values
result_df['Team 2'] = data.loc[train_size:, 'Team 2'].values
result_df['Map Name'] = data.loc[train_size:, 'Map Name'].values

# Show results for analysis
print(result_df[['Team 1', 'Team 2', 'Map Name', 'Actual Score Difference', 'Predicted Score Difference']])

predicted_winner = (y_pred > 0).astype(int)  # If predicted score difference > 0, Team 1 wins (1), else Team 2 wins (0)
actual_winner = (y_test > 0).astype(int)     # If actual score difference > 0, Team 1 wins (1), else Team 2 wins (0)

accuracy = np.mean(predicted_winner == actual_winner)
print(f'Accuracy of Predicted Winners: {accuracy * 100:.2f}%')

result_df.to_csv('data/predictions.csv', index=False)

# --- VISUALIZATION ---
# Scatter plot of actual vs predicted score difference
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # line of perfect prediction
plt.title('Actual vs Predicted Score Difference')
plt.xlabel('Actual Score Difference')
plt.ylabel('Predicted Score Difference')

# Ensure the plot is not empty and then save
plt.tight_layout()  # Adjust layout to avoid cut-off text and axes
plt.savefig("images/actual_vs_predicted_score_difference.png", dpi=300, bbox_inches='tight')
plt.close()

# Histogram of residuals (errors)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='green')
plt.title('Distribution of Residuals')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')

# Ensure the plot is not empty and then save
plt.tight_layout()  # Adjust layout to avoid cut-off text and axes
plt.savefig("images/accuracy_of_predicted_winners.png", dpi=300, bbox_inches='tight')
plt.close()

# Get feature importances
feature_importances = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# Calculate correlations
correlation_matrix = X.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

def categorize_difference(value):
    """
    Categorizes the predicted score difference range.
    """
    if abs(value) <= 2:
        return "Very Close (<=2)"
    elif abs(value) <= 5:
        return "Close (3-5)"
    else:
        return "Large (>5)"

# Add a column for the difference range category based on predicted score difference
result_df['Difference Range'] = result_df['Predicted Score Difference'].apply(categorize_difference)

# Calculate accuracy for each range
accuracy_by_range = result_df.groupby('Difference Range').apply(
    lambda group: np.mean((group['Predicted Score Difference'] > 0).astype(int) ==
                          (group['Actual Score Difference'] > 0).astype(int))
)


# Plot accuracy by range
accuracy_by_range.plot(kind='bar', figsize=(8, 5), color='skyblue', edgecolor='black')
plt.title('Accuracy by Score Difference Range')
plt.ylabel('Accuracy (%)')
plt.xlabel('Score Difference Range')
plt.tight_layout()
plt.savefig("images/accuracy_by_range.png", dpi=300, bbox_inches='tight')
plt.close()

selected_features = ['Recent Form Difference', 'Win Rate Difference']
for feature in selected_features:
    plt.figure(figsize=(8, 5))
    plt.scatter(result_df[feature], residuals, alpha=0.7, color='orange')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Residuals vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(f"images/residuals_vs_{feature}.png", dpi=300, bbox_inches='tight')
    plt.close()
