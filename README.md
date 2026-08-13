# Thiranex Task 2 - Football Machine Learning Data Loading & Preprocessing

A Python Data Science & Machine Learning project for **Task 2 Internship Assignment**.

## 📌 Project Overview
This project loads, inspects, cleans, and preprocesses the Premier League football machine learning raw dataset (`football_ml_raw_dataset.csv`) to prepare it for predictive modeling.

## 📁 Repository Structure
```text
├── football_ml_raw_dataset.csv    # Raw dataset containing match statistics
├── football_ml_analysis.py       # Python script for loading, cleaning & preprocessing data
└── README.md                     # Project documentation
```

## 🛠️ Data Science & ML Workflow
1. **Load Data**: Import raw dataset into Pandas DataFrame.
2. **Data Inspection**: Check dataset dimensions, data types, summary stats, missing values, and target variable distribution (`FTR` - Full Time Result).
3. **Data Preprocessing**:
   - Strip whitespace from categorical columns (`HomeTeam`, `AwayTeam`, etc.).
   - Impute missing numerical values (`HS`, `AS`, `HST`, `AST`, `HC`, `AC`) using column medians.
   - Prepare structured DataFrame ready for machine learning feature engineering and model training.

## 🚀 How to Run
Ensure you have `pandas` and `numpy` installed:

```bash
pip install pandas numpy
python football_ml_analysis.py
```
