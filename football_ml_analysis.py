import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("football_ml_raw_dataset.csv")

# 1. Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='mixed')

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Standardize team names
team_mapping = {
    "Manchester City": "Man City",
    "Man united": "Man United"
}
df["HomeTeam"] = df["HomeTeam"].replace(team_mapping)
df["AwayTeam"] = df["AwayTeam"].replace(team_mapping)

# 4. Fill missing numerical values with column median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ============================================================
# STEP 4: OUTLIER DETECTION AND HANDLING
# ============================================================

# Select numerical statistics columns to inspect
stats_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF']

# Create boxplot visualization
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[stats_cols])
plt.title("Boxplot of Selected Football Statistics")
plt.xlabel("Match Statistics")
plt.ylabel("Values")
plt.tight_layout()
plt.savefig("outliers_boxplot.png")
plt.close()
print("Saved boxplot visualization to 'outliers_boxplot.png'")

# Detect outliers using the IQR method
print("\n--- Outlier Analysis using IQR Method ---")
for col in stats_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
    
    print(f"\n[{col}] Q1: {Q1:.2f} | Q3: {Q3:.2f} | IQR: {IQR:.2f}")
    print(f"  Limits -> Lower: {lower_limit:.2f} | Upper: {upper_limit:.2f}")
    print(f"  Number of potential outliers: {len(outliers)}")
    
    if len(outliers) > 0:
        print("  Sample outlier matches:")
        print(outliers[['Date', 'HomeTeam', 'AwayTeam', col]].head())

# Check for impossible/invalid negative values
negative_values = df[(df[stats_cols] < 0).any(axis=1)]
if len(negative_values) > 0:
    print(f"\nFound {len(negative_values)} invalid negative records. Removing them...")
    df = df[(df[stats_cols] >= 0).all(axis=1)]
else:
    print("\nNo impossible/negative values found.")

# Summary message
print("\nPotential outliers were identified using the IQR method. The unusual values were reviewed rather than automatically removed because extreme football statistics can represent genuine match events.")

# Save cleaned dataset
df.to_csv("football_ml_cleaned_dataset.csv", index=False)
