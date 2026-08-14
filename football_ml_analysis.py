import pandas as pd

# Load dataset and clean date/duplicates
df = pd.read_csv("football_ml_raw_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df = df.sort_values('Date').drop_duplicates().reset_index(drop=True)

# Standardize team names
team_mapping = {"Manchester City": "Man City", "Man united": "Man United"}
df["HomeTeam"] = df["HomeTeam"].replace(team_mapping)
df["AwayTeam"] = df["AwayTeam"].replace(team_mapping)

# Fill missing numerical values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ============================================================
# RECENT FORM FEATURE ENGINEERING (LAST 5 MATCHES)
# ============================================================

# Calculate match points
df['HomePoints'] = df['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
df['AwayPoints'] = df['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

# Get recent form (last 5 matches) for a team before a match index
def get_recent_form(team, current_idx):
    past = df.iloc[:current_idx]
    
    home_g = past[past['HomeTeam'] == team][['Date', 'HomePoints', 'FTHG', 'FTAG', 'HS', 'HST']]
    home_g.columns = ['Date', 'Pts', 'GF', 'GA', 'Shots', 'SoT']
    
    away_g = past[past['AwayTeam'] == team][['Date', 'AwayPoints', 'FTAG', 'FTHG', 'AS', 'AST']]
    away_g.columns = ['Date', 'Pts', 'GF', 'GA', 'Shots', 'SoT']
    
    last5 = pd.concat([home_g, away_g]).sort_values('Date').tail(5)
    if len(last5) == 0:
        return 0, 0.0, 0.0, 0.0
    
    return (
        int(last5['Pts'].sum()),
        round(float(last5['GF'].mean()), 2),
        round(float(last5['Shots'].mean()), 2),
        round(float(last5['SoT'].mean()), 2)
    )

# Calculate features for Home and Away teams
home_form = [get_recent_form(df.loc[i, 'HomeTeam'], i) for i in range(len(df))]
away_form = [get_recent_form(df.loc[i, 'AwayTeam'], i) for i in range(len(df))]

df[['HomeRecentPoints', 'HomeAvgGoals', 'HomeAvgShots', 'HomeAvgShotsOnTarget']] = home_form
df[['AwayRecentPoints', 'AwayAvgGoals', 'AwayAvgShots', 'AwayAvgShotsOnTarget']] = away_form

# Cleanup temporary calculation columns
df.drop(columns=['HomePoints', 'AwayPoints'], inplace=True)

# ============================================================
# FEATURE MATRIX (X) AND TARGET VARIABLE (y)
# ============================================================

features = [
    "HomeRecentPoints",
    "AwayRecentPoints",
    "HomeAvgGoals",
    "AwayAvgGoals",
    "HomeAvgShots",
    "AwayAvgShots",
    "HomeAvgShotsOnTarget",
    "AwayAvgShotsOnTarget"
]

X = df[features]
y = df["FTR"]

# Save cleaned dataset with engineered features
df.to_csv("football_ml_cleaned_dataset.csv", index=False)

print("Data cleaning, feature engineering, and X/y selection completed!")
print("Feature matrix X shape:", X.shape)
print("Target vector y shape:", y.shape)

print("\n--- First 5 rows of X ---")
print(X.head())

print("\n--- First 5 rows of y ---")
print(y.head())

# ============================================================
# STEP 6.4: TRAIN / TEST SPLIT (CHRONOLOGICAL SPLIT)
# ============================================================

# Step 1: Ensure the dataset is sorted by Date column (past to recent matches)
df = df.sort_values("Date").reset_index(drop=True)

# Re-assign feature matrix X and target y after sorting
X = df[features]
y = df["FTR"]

# Step 2: Split into 80% training (older matches) and 20% testing (newer matches)
# Set shuffle=False to preserve chronological order for football prediction
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False
)

print("\n--- Step 6.4: Train/Test Split Completed ---")
print("Total matches:", len(df))
print(f"Training set (older 80%): X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
print(f"Testing set  (newer 20%): X_test shape  = {X_test.shape}, y_test shape  = {y_test.shape}")

# ============================================================
# STEP 6.5: MODEL CREATION (DECISION TREE CLASSIFIER)
# ============================================================

from sklearn.tree import DecisionTreeClassifier

# Initialize Decision Tree Classifier model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

print("\n--- Model Initialized ---")
print("Model configuration:", model)

# ============================================================
# STEP 6.6: MODEL TRAINING
# ============================================================

# Train the Decision Tree Classifier model on older matches (X_train, y_train)
model.fit(X_train, y_train)

print("\n--- Model Training Completed ---")
print("The Decision Tree Classifier model has been trained successfully!")

# ============================================================
# STEP 6.7: MAKE PREDICTIONS
# ============================================================

# Generate predictions on the test dataset (newer 20% matches)
predictions = model.predict(X_test)

print("\n--- Predictions Completed ---")
print("Total test predictions made:", len(predictions))
print("First 10 predictions:", predictions[:10])

# ============================================================
# STEP 6.8: MODEL ACCURACY & COMPARISON
# ============================================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("\nDecision Tree Accuracy:", accuracy)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n--- First 10 Test Matches: Actual vs Predicted ---")
print(results.head(10))

# ============================================================
# STEP 6.9: CLASSIFICATION REPORT
# ============================================================

from sklearn.metrics import classification_report

print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))








