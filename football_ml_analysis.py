import pandas as pd

# Load raw dataset
df = pd.read_csv("football_ml_raw_dataset.csv")

# 1. Clean date, duplicates, team names & missing values
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df = df.sort_values('Date').drop_duplicates().reset_index(drop=True)

team_map = {"Manchester City": "Man City", "Man united": "Man United"}
df["HomeTeam"] = df["HomeTeam"].replace(team_map)
df["AwayTeam"] = df["AwayTeam"].replace(team_map)

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 2. Assign match points
df['H_Pts'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
df['A_Pts'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})

# 3. Helper to get recent performance (last 5 matches)
def get_team_form(team, idx):
    prev = df.iloc[:idx]
    home_m = prev[prev['HomeTeam'] == team].rename(columns={'H_Pts': 'pts', 'FTHG': 'gf', 'FTAG': 'ga', 'HS': 'shots', 'HST': 'sot'})
    away_m = prev[prev['AwayTeam'] == team].rename(columns={'A_Pts': 'pts', 'FTAG': 'gf', 'FTHG': 'ga', 'AS': 'shots', 'AST': 'sot'})
    last5 = pd.concat([home_m, away_m]).sort_values('Date').tail(5)
    
    if last5.empty:
        return [0, 0, 0, 0.0, 0.0]
    return [int(last5['pts'].sum()), int(last5['gf'].sum()), int(last5['ga'].sum()), round(last5['shots'].mean(), 2), round(last5['sot'].mean(), 2)]

# 4. Calculate recent form for Home and Away teams
form_cols = ['Pts_L5', 'GF_L5', 'GA_L5', 'Shots_L5', 'SoT_L5']
df[['Home_' + c for c in form_cols]] = [get_team_form(row['HomeTeam'], i) for i, row in df.iterrows()]
df[['Away_' + c for c in form_cols]] = [get_team_form(row['AwayTeam'], i) for i, row in df.iterrows()]

# Drop temporary columns and save
df.drop(columns=['H_Pts', 'A_Pts'], inplace=True)
df.to_csv("football_ml_cleaned_dataset.csv", index=False)

print("Data cleaning & feature engineering completed! Cleaned dataset shape:", df.shape)
