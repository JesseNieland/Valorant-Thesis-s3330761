import pandas as pd

# Load dataset
file_path = 'vct_2024_matches_cleaned_final.csv'
df = pd.read_csv(file_path)

# Check data types of each column
print("Data Types of Each Column:")
print(df.dtypes)

# Check the types of entries in each column by examining the first non-null entry
print("\nData Types of First Non-Null Entry in Each Column:")
for col in df.columns:
    first_entry = df[col].dropna().iloc[0]  # Fetch the first non-null entry in each column
    print(f"{col}: {type(first_entry)}")  # Print the type of this entry
