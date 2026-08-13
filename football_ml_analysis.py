import pandas as pd

# Load the dataset
df = pd.read_csv("football_ml_raw_dataset.csv")

# 1. Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='mixed')

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Strip whitespace from HomeTeam and AwayTeam
df["HomeTeam"] = df["HomeTeam"].str.strip()
df["AwayTeam"] = df["AwayTeam"].str.strip()

# 4. Standardize team names
team_mapping = {
    "Manchester City": "Man City",
    "Man united": "Man United"
}
df["HomeTeam"] = df["HomeTeam"].replace(team_mapping)
df["AwayTeam"] = df["AwayTeam"].replace(team_mapping)

# 5. Fill missing numerical values with column median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Save cleaned data
df.to_csv("football_ml_cleaned_dataset.csv", index=False)

print("Data cleaning completed successfully!")
print("Cleaned shape:", df.shape)
print("Remaining missing values:\n", df.isnull().sum())
