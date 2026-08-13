import pandas as pd

# Load the dataset
df = pd.read_csv("football_ml_raw_dataset.csv")

# 1. Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='mixed')

# 2. Sort chronologically and remove duplicate rows
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates().reset_index(drop=True)

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
# RECENT FORM FEATURE ENGINEERING (LAST 5 MATCHES)
# ============================================================

# Calculate points earned in each match
df['HomePoints'] = df['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
df['AwayPoints'] = df['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

# Simple helper function to get team performance in their last 5 previous matches
def get_team_recent_form(team, current_row_index):
    # Select all matches played before the current match
    past_matches = df.iloc[:current_row_index]
    
    # Filter matches where the team played as Home or Away
    home_games = past_matches[past_matches['HomeTeam'] == team]
    away_games = past_matches[past_matches['AwayTeam'] == team]
    
    # Select relevant columns and rename them to match uniform names
    home_stats = home_games[['Date', 'HomePoints', 'FTHG', 'FTAG', 'HS', 'HST']].copy()
    home_stats.columns = ['Date', 'Pts', 'GF', 'GA', 'Shots', 'SoT']
    
    away_stats = away_games[['Date', 'AwayPoints', 'FTAG', 'FTHG', 'AS', 'AST']].copy()
    away_stats.columns = ['Date', 'Pts', 'GF', 'GA', 'Shots', 'SoT']
    
    # Combine home and away matches, sort by date, and pick the last 5 matches
    all_team_matches = pd.concat([home_stats, away_stats]).sort_values('Date')
    last_5_matches = all_team_matches.tail(5)
    
    # If the team has no prior matches, return zeros
    if len(last_5_matches) == 0:
        return 0, 0, 0, 0.0, 0.0
    
    total_pts = int(last_5_matches['Pts'].sum())
    total_gf = int(last_5_matches['GF'].sum())
    total_ga = int(last_5_matches['GA'].sum())
    avg_shots = round(float(last_5_matches['Shots'].mean()), 2)
    avg_sot = round(float(last_5_matches['SoT'].mean()), 2)
    
    return total_pts, total_gf, total_ga, avg_shots, avg_sot

# Lists to store calculated features
home_pts, home_gf, home_ga, home_shots, home_sot = [], [], [], [], []
away_pts, away_gf, away_ga, away_shots, away_sot = [], [], [], [], []

# Loop through each match to calculate team form before the game
for i in range(len(df)):
    hp, hgf, hga, hs, hsot = get_team_recent_form(df.loc[i, 'HomeTeam'], i)
    ap, agf, aga, aws, asot = get_team_recent_form(df.loc[i, 'AwayTeam'], i)
    
    home_pts.append(hp)
    home_gf.append(hgf)
    home_ga.append(hga)
    home_shots.append(hs)
    home_sot.append(hsot)
    
    away_pts.append(ap)
    away_gf.append(agf)
    away_ga.append(aga)
    away_shots.append(aws)
    away_sot.append(asot)

# Add new columns to DataFrame
df['Home_Pts_L5'] = home_pts
df['Home_GF_L5'] = home_gf
df['Home_GA_L5'] = home_ga
df['Home_Shots_L5'] = home_shots
df['Home_SoT_L5'] = home_sot

df['Away_Pts_L5'] = away_pts
df['Away_GF_L5'] = away_gf
df['Away_GA_L5'] = away_ga
df['Away_Shots_L5'] = away_shots
df['Away_SoT_L5'] = away_sot

# Remove temporary calculation columns
df.drop(columns=['HomePoints', 'AwayPoints'], inplace=True)

# Save cleaned dataset with engineered features
df.to_csv("football_ml_cleaned_dataset.csv", index=False)

print("Data cleaning & feature engineering completed successfully!")
print("Cleaned Dataset Shape:", df.shape)
