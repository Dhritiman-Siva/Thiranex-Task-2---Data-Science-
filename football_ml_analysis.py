import pandas as pd

# Load the dataset
df = pd.read_csv("football_ml_raw_dataset.csv")

# Display basic information
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Summary ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())
