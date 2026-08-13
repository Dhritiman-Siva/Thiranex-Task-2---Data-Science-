import pandas as pd

# Load raw dataset
df = pd.read_csv("football_ml_raw_dataset.csv")

# 1. Convert Date to datetime format and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df = df.sort_values('Date').reset_index(drop=True)

# 2. Remove duplicate rows and reset index
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
# CALCULATING RECENT FORM (LAST 5 MATCHES)
# ============================================================

# Calculate points earned in each match (Home: 3 for Win, 1 for Draw; Away: 3 for Win, 1 for Draw)
df['HomePoints'] = df['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
df['AwayPoints'] = df['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

# Helper function to get team performance in their last 5 previous matches
def get_recent_stats(team, current_index):
    prev_matches = []
    
    # Look back at all matches played before current_index
    for i in range(current_index):
        row = df.iloc[i]
        if row['HomeTeam'] == team:
            prev_matches.append({
                'pts': row['HomePoints'],
                'gf': row['FTHG'],
                'ga': row['FTAG'],
                'shots': row['HS'],
                'sot': row['HST']
            })
        elif row['AwayTeam'] == team:
            prev_matches.append({
                'pts': row['AwayPoints'],
                'gf': row['FTAG'],
                'ga': row['FTHG'],
                'shots': row['AS'],
                'sot': row['AST']
            })
    
    # Get last 5 matches
    last_5 = prev_matches[-5:]
    
    if len(last_5) == 0:
        return 0, 0, 0, 0.0, 0.0
    
    pts = sum(m['pts'] for m in last_5)
    gf = sum(m['gf'] for m in last_5)
    ga = sum(m['ga'] for m in last_5)
    avg_shots = round(sum(m['shots'] for m in last_5) / len(last_5), 2)
    avg_sot = round(sum(m['sot'] for m in last_5) / len(last_5), 2)
    
    return pts, gf, ga, avg_shots, avg_sot

# Lists to hold engineered features
home_pts, home_gf, home_ga, home_shots, home_sot = [], [], [], [], []
away_pts, away_gf, away_ga, away_shots, away_sot = [], [], [], [], []

# Loop through every match to calculate recent form before the match
for i in range(len(df)):
    hp, hgf, hga, hs, hsot = get_recent_stats(df.iloc[i]['HomeTeam'], i)
    ap, agf, aga, aws, asot = get_recent_stats(df.iloc[i]['AwayTeam'], i)
    
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

# Add recent form features to DataFrame
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

# Clean temporary calculation columns
df.drop(columns=['HomePoints', 'AwayPoints'], inplace=True)

# Save updated dataset with recent form features
df.to_csv("football_ml_cleaned_dataset.csv", index=False)

print("Recent form features calculated successfully!")
print("\nSample Preview of Engineered Features (First 10 Rows):")
preview_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Home_Pts_L5', 'Home_GF_L5', 'Away_Pts_L5', 'Away_GF_L5']
print(df[preview_cols].head(10))
