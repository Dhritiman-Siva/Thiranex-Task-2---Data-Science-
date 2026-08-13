# Football Machine Learning Project - Task 2
# Load, Preprocess, and Analyze football_ml_raw_dataset.csv

import pandas as pd
import numpy as np
import os

def load_data(filepath="football_ml_raw_dataset.csv"):
    """
    Loads football ML dataset from a CSV file.
    """
    if not os.path.exists(filepath):
        # Fallback to absolute/relative check if run from root directory
        alt_path = os.path.join("Task2", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at '{filepath}' or '{alt_path}'")

    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully!")
    return df

def inspect_data(df):
    """
    Displays dataset summary statistics, shape, columns, missing values, and duplicates.
    """
    print("\n" + "="*60)
    print(" DATASET OVERVIEW & INSPECTION ")
    print("="*60)
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Column Names & Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values Summary ---")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print(missing_cols)
    else:
        print("No missing values found.")

    print("\n--- Duplicate Rows ---")
    print(f"Duplicates: {df.duplicated().sum()}")

    print("\n--- Target Variable Distribution (FTR: Full Time Result) ---")
    if 'FTR' in df.columns:
        print(df['FTR'].value_counts(dropna=False))

def preprocess_data(df):
    """
    Cleans dataset and fills missing numerical values with median.
    """
    df_clean = df.copy()
    
    # Strip whitespace from string columns
    str_cols = df_clean.select_dtypes(include=['object']).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Fill missing values in numerical columns with column median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median: {median_val}")

    return df_clean

def main():
    # 1. Load Dataset
    df = load_data()
    
    # 2. Inspect Data
    inspect_data(df)
    
    # 3. Clean & Preprocess
    print("\n" + "="*60)
    print(" PREPROCESSING ")
    print("="*60)
    df_clean = preprocess_data(df)
    
    print(f"\nProcessed Dataset Shape: {df_clean.shape}")
    print("Data loading and initial inspection complete!")

if __name__ == "__main__":
    main()
