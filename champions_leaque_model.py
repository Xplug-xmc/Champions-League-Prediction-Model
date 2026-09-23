# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import re
import json
import joblib
import uuid
from pathlib import Path
from collections import deque


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# LOAD DATASET
df = pd.read_csv("data/raw/champions_league_2024_25_fbref_raw.csv")


# DATASET SHAPE
print("Dataset shape:", df.shape)


# DATASET COLUMNS
print("\nColumns:")
print(df.columns.tolist())


# FIRST 5 ROWS
print("\nFirst 5 rows:")
print(df.head())


# MISSING VALUES
print("\nMissing values:")
print(df.isnull().sum())


# MATCH ROUNDS
print("\nRounds:")
print(df["round"].value_counts())


# INSPECT SCORE FORMATS
print("\nUnique score formats:")
print(df["score"].unique())


# INSPECT KNOCKOUT MATCH NOTES
print("\nMatches with notes:")
print(
    df[df["notes"].notna()][
        ["round", "date", "home_team", "score", "away_team", "notes"]
    ].to_string(index=False)
)


# EXTRACT HOME AND AWAY GOALS
def extract_goals(score):
    match = re.search(r"(\d+)\s*–\s*(\d+)", str(score))
    
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None, None


df[["home_goals", "away_goals"]] = df["score"].apply(
    lambda x: pd.Series(extract_goals(x))
)


# CREATE MATCH RESULT
def get_result(row):
    if row["home_goals"] > row["away_goals"]:
        return "H"
    elif row["home_goals"] < row["away_goals"]:
        return "A"
    else:
        return "D"


df["result"] = df.apply(get_result, axis=1)


print(df[
    ["date", "home_team", "score", "away_team",
     "home_goals", "away_goals", "result"]
].tail(20))


# CHECK PENALTY SHOOT-OUT MATCHES
penalty_matches = df[
    df["score"].str.contains(r"\(", na=False)
]

print(
    penalty_matches[
        ["date", "home_team", "score", "away_team",
         "home_goals", "away_goals", "result", "notes"]
    ].to_string(index=False)
)


# CHECK FOR MISSING GOALS AND RESULTS
print("\nMissing Home Goals:")
print(df["home_goals"].isnull().sum())

print("\nMissing Away Goals:")
print(df["away_goals"].isnull().sum())

print("\nMissing Results:")
print(df["result"].isnull().sum())


# CHECK RESULT DISTRIBUTION
print("\nResult Distribution:")
print(df["result"].value_counts())


# CHECK THE CLEAN DATASET
print("\nClean Dataset Preview:")

print(
    df[
        [
            "season",
            "round",
            "date",
            "home_team",
            "home_goals",
            "away_goals",
            "away_team",
            "result"
        ]
    ].head(10)
)


# CREATE CLEAN 2024/25 DATASET
clean_df = df[
    [
        "season",
        "round",
        "date",
        "home_team",
        "home_goals",
        "away_goals",
        "away_team",
        "result"
    ]
].copy()


# CHECK CLEAN DATASET
print("Clean dataset shape:", clean_df.shape)

print("\nClean dataset columns:")
print(clean_df.columns.tolist())

print("\nFirst 10 rows:")
print(clean_df.head(10))


# SAVE CLEAN DATASET
clean_df.to_csv(
    "champions_league_2024_25_clean.csv",
    index=False
)


# CLEAN TEAM NAMES
def clean_team_name(team):
    # Convert the value to text
    team = str(team)

    # Remove country code at the beginning
    team = re.sub(r"^[a-z]{2,3}\s+", "", team)

    # Remove extra spaces
    team = team.strip()

    return team


# APPLY TEAM NAME CLEANING
clean_df["home_team"] = clean_df["home_team"].apply(
    clean_team_name
)

clean_df["away_team"] = clean_df["away_team"].apply(
    clean_team_name
)


# CHECK CLEAN TEAM NAMES
print("Home Teams:")
print(clean_df["home_team"].unique())

print("\nAway Teams:")
print(clean_df["away_team"].unique())


# RELOAD RAW DATASET
df = pd.read_csv("data/raw/champions_league_2024_25_fbref_raw.csv")


# EXTRACT HOME AND AWAY GOALS

def extract_goals(score):
    # Convert the score to text
    score = str(score)

    # Find the two match-goal numbers
    match = re.search(r"(\d+)\s*–\s*(\d+)", score)

    # If a match is found, return the two goals
    if match:
        return int(match.group(1)), int(match.group(2))

    # Return missing values if no score is found
    return None, None


df[["home_goals", "away_goals"]] = df["score"].apply(
    lambda x: pd.Series(extract_goals(x))
)


# CREATE MATCH RESULT
def get_result(row):
    # Home team scored more goals
    if row["home_goals"] > row["away_goals"]:
        return "H"

    # Away team scored more goals
    elif row["home_goals"] < row["away_goals"]:
        return "A"

    # Both teams scored the same number of goals
    else:
        return "D"


df["result"] = df.apply(get_result, axis=1)


# CREATE CLEAN DATASET
clean_df = df[
    [
        "season",
        "round",
        "date",
        "home_team",
        "home_goals",
        "away_goals",
        "away_team",
        "result"
    ]
].copy()


# CLEAN TEAM NAMES
def clean_team_name(team):
    # Convert the team name to text
    team = str(team)

    # Remove the country code at the beginning
    team = re.sub(r"^[a-z]{2,3}\s+", "", team)

    # Remove unnecessary spaces
    team = team.strip()

    return team


# APPLY TEAM NAME CLEANING
clean_df["home_team"] = clean_df["home_team"].apply(
    clean_team_name
)

clean_df["away_team"] = clean_df["away_team"].apply(
    clean_team_name
)


# CHECK TEAM NAMES
print("\nHome Teams:")
print(sorted(clean_df["home_team"].unique()))

print("\nAway Teams:")
print(sorted(clean_df["away_team"].unique()))


# SAVE CLEAN 2024/25 DATASET
clean_df.to_csv(
    "champions_league_2024_25_clean.csv",
    index=False
)

print("Clean 2024/25 dataset saved successfully.")

print("\nShape:", clean_df.shape)

print("\nColumns:")
print(clean_df.columns.tolist())

print("\n")

# LOAD 2023/24 RAW DATASET
df_2023_24 = pd.read_csv("data/raw/champions_league_2023_24_fbref_raw.csv")


# CHECK DATASET SHAPE
print("Dataset shape:", df_2023_24.shape)


# CHECK COLUMNS
print("\nColumns:")
print(df_2023_24.columns.tolist())


# CHECK FIRST 5 ROWS
print("\nFirst 5 rows:")
print(df_2023_24.head())


# CHECK MISSING VALUES
print("\nMissing values:")
print(df_2023_24.isnull().sum())


# CHECK MATCH ROUNDS
print("\nRounds:")
print(df_2023_24["round"].value_counts())


# EXTRACT HOME AND AWAY GOALS
def extract_goals(score):
    # Convert the score to text
    score = str(score)

    # Find the two match-goal numbers
    match = re.search(r"(\d+)\s*–\s*(\d+)", score)

    # Return the goals if a match is found
    if match:
        return int(match.group(1)), int(match.group(2))

    # Return missing values if no score is found
    return None, None


df_2023_24[["home_goals", "away_goals"]] = df_2023_24["score"].apply(
    lambda x: pd.Series(extract_goals(x))
)


# CREATE MATCH RESULT
def get_result(row):
    # Home team won
    if row["home_goals"] > row["away_goals"]:
        return "H"

    # Away team won
    elif row["home_goals"] < row["away_goals"]:
        return "A"

    # Match was drawn
    else:
        return "D"


df_2023_24["result"] = df_2023_24.apply(
    get_result,
    axis=1
)


# INSPECT SPECIAL SCORE FORMATS
print("\nScore formats containing parentheses:")

print(
    df_2023_24[
        df_2023_24["score"].str.contains(r"\(", na=False)
    ][
        [
            "date",
            "home",
            "score",
            "away",
            "home_goals",
            "away_goals",
            "notes"
        ]
    ].to_string(index=False)
)

print("\n")

# STANDARDIZE TEAM COLUMN NAMES
df_2023_24 = df_2023_24.rename(
    columns={
        "home": "home_team",
        "away": "away_team"
    }
)


# CHECK UPDATED COLUMN NAMES
print(df_2023_24.columns.tolist())


print("\n")

# CLEAN TEAM NAMES
def clean_team_name(team):
    # Convert the value to text
    team = str(team)

    # Remove the country code at the beginning
    team = re.sub(r"^[a-z]{2,3}\s+", "", team)

    # Remove extra spaces
    team = team.strip()

    return team


# APPLY CLEANING TO HOME TEAMS
df_2023_24["home_team"] = df_2023_24["home_team"].apply(
    clean_team_name
)


# APPLY CLEANING TO AWAY TEAMS
df_2023_24["away_team"] = df_2023_24["away_team"].apply(
    clean_team_name
)


# CHECK TEAM NAMES
print("\nHome Teams:")
print(sorted(df_2023_24["home_team"].unique()))

print("\nAway Teams:")
print(sorted(df_2023_24["away_team"].unique()))



print("\n")

# CLEAN 2023/24 HOME TEAM NAMES
def clean_home_team_name(team):
    # Convert the value to text
    team = str(team)

    # Remove the country code at the end
    team = re.sub(r"\s+[a-z]{2,3}$", "", team)

    # Remove unnecessary spaces
    team = team.strip()

    return team


# APPLY CLEANING TO HOME TEAMS
df_2023_24["home_team"] = df_2023_24["home_team"].apply(
    clean_home_team_name
)


# CHECK HOME TEAM NAMES
print("\nHome Teams:")
print(sorted(df_2023_24["home_team"].unique()))

print("\n")

# CREATE MATCH RESULT
def get_result(row):
    # Home team scored more goals
    if row["home_goals"] > row["away_goals"]:
        return "H"

    # Away team scored more goals
    elif row["home_goals"] < row["away_goals"]:
        return "A"

    # Both teams scored the same number of goals
    else:
        return "D"


    # CHECK RESULT DISTRIBUTION
print("\nResult Distribution:")
print(df_2023_24["result"].value_counts())

print("\nMissing Results:")
print(df_2023_24["result"].isnull().sum())



print("\n")

# CREATE CLEAN 2023/24 DATASET
clean_2023_24 = df_2023_24[
    [
        "season",
        "round",
        "date",
        "home_team",
        "home_goals",
        "away_goals",
        "away_team",
        "result"
    ]
].copy()


# CHECK CLEAN 2023/24 DATASET
print("Clean dataset shape:", clean_2023_24.shape)

print("\nColumns:")
print(clean_2023_24.columns.tolist())

print("\nFirst 10 rows:")
print(clean_2023_24.head(10))


# SAVE CLEAN 2023/24 DATASET
clean_2023_24.to_csv(
    "champions_league_2023_24_clean.csv",
    index=False
)

print("Clean 2023/24 dataset saved successfully.")


print("\n")

# LOAD CLEAN 2023/24 DATASET
clean_2023_24 = pd.read_csv(
    "champions_league_2023_24_clean.csv"
)


# LOAD CLEAN 2024/25 DATASET
clean_2024_25 = pd.read_csv(
    "champions_league_2024_25_clean.csv"
)


# CHECK DATASET SHAPES
print("2023/24 Shape:", clean_2023_24.shape)
print("2024/25 Shape:", clean_2024_25.shape)


# CHECK COLUMN NAMES
print("\n2023/24 Columns:")
print(clean_2023_24.columns.tolist())

print("\n2024/25 Columns:")
print(clean_2024_25.columns.tolist())


# CHECK WHETHER THE COLUMN STRUCTURES ARE IDENTICAL
print(
    "\nColumns identical:",
    list(clean_2023_24.columns) == list(clean_2024_25.columns)
)


# CHECK RESULT VALUES
print("\n2023/24 Results:")
print(sorted(clean_2023_24["result"].unique()))

print("\n2024/25 Results:")
print(sorted(clean_2024_25["result"].unique()))


print("\n")

# COMBINE 2023/24 AND 2024/25

combined_df = pd.concat(
    [clean_2023_24, clean_2024_25],
    ignore_index=True
)


# CHECK COMBINED DATASET
print("Combined Dataset Shape:", combined_df.shape)

print("\nColumns:")
print(combined_df.columns.tolist())

print("\nFirst 5 rows:")
print(combined_df.head())

print("\nLast 5 rows:")
print(combined_df.tail())


print("\n")

# CHECK FOR DUPLICATE MATCHES
duplicates = combined_df.duplicated(
    subset=["season", "date", "home_team", "away_team"],
    keep=False
)

print("Number of duplicate rows:", duplicates.sum())


# CHECK DATE RANGE
print("Earliest match:", combined_df["date"].min())
print("Latest match:", combined_df["date"].max())


# CHECK MATCH COUNT BY SEASON
print("\nMatches by season:")
print(combined_df["season"].value_counts().sort_index())


print("\n")


# CLEAN TEAM NAMES FROM BOTH FBREF FORMATS
def clean_team_name(team):
    # Convert the value to text
    team = str(team)

    # Remove country code at the beginning OR at the end
    team = re.sub(
        r"^[a-z]{2,3}\s+|\s+[a-z]{2,3}$",
        "",
        team
    )

    # Remove unnecessary spaces
    team = team.strip()

    return team


# CREATE A FUNCTION TO EXTRACT GOALS
def extract_goals(score):
    # Convert the score to text
    score = str(score)

    # Find the two match-goal numbers
    match = re.search(r"(\d+)\s*–\s*(\d+)", score)

    # Return the goals if found
    if match:
        return int(match.group(1)), int(match.group(2))

    # Return missing values if no score is found
    return None, None


# CREATE A FUNCTION TO ASSIGN MATCH RESULT
def get_result(row):
    # Home team won
    if row["home_goals"] > row["away_goals"]:
        return "H"

    # Away team won
    elif row["home_goals"] < row["away_goals"]:
        return "A"

    # Match was drawn
    else:
        return "D"


    # CREATE A REUSABLE SEASON-CLEANING FUNCTION
# Clean Season Data
def clean_season_data(raw_df):

    # Make a copy of the raw data
    df = raw_df.copy()

    # Standardize column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Rename team columns
    df = df.rename(
        columns={
            "home": "home_team",
            "away": "away_team"
        }
    )

    # Extract home and away goals from the score
    df[["home_goals", "away_goals"]] = df["score"].apply(
        lambda x: pd.Series(extract_goals(x))
    )

    # Calculate match result
    df["result"] = df.apply(get_result, axis=1)

    # Clean team names
    df["home_team"] = df["home_team"].apply(clean_team_name)
    df["away_team"] = df["away_team"].apply(clean_team_name)

    # Select final columns
    clean_df = df[
        [
            "season",
            "round",
            "date",
            "home_team",
            "home_goals",
            "away_goals",
            "away_team",
            "result"
        ]
    ].copy()

    return clean_df


# TEST THE REUSABLE FUNCTION WITH 2023/24
test_2023_24 = clean_season_data(
    df_2023_24
)

print("Cleaned 2023/24 Shape:")
print(test_2023_24.shape)

print("\nCleaned 2023/24 Columns:")
print(test_2023_24.columns.tolist())

print("\nFirst 5 Rows:")
print(test_2023_24.head())


# TEST THE REUSABLE FUNCTION WITH 2024/25
test_2024_25 = clean_season_data(
    df
)

print("Cleaned 2024/25 Shape:")
print(test_2024_25.shape)

print("\nCleaned 2024/25 Columns:")
print(test_2024_25.columns.tolist())

print("\nFirst 5 Rows:")
print(test_2024_25.head())


# TEST CLEANING ON 2023/24 AGAIN
test_2023_24 = clean_season_data(
    df_2023_24
)

print("\n Cleaned 2023/24 Shape:")
print(test_2023_24.shape)

print("\nFirst 5 Rows:")
print(test_2023_24.head())


print("\n")


# VALIDATE 2023/24 CLEAN DATA
print("2023/24 Missing Values:")
print(test_2023_24.isnull().sum())

print("\n2023/24 Result Distribution:")
print(test_2023_24["result"].value_counts())

print("\n2023/24 Duplicate Matches:")
print(
    test_2023_24.duplicated(
        subset=["season", "date", "home_team", "away_team"]
    ).sum()
)


# VALIDATE 2024/25 CLEAN DATA
print("\n2024/25 Missing Values:")
print(test_2024_25.isnull().sum())

print("\n2024/25 Result Distribution:")
print(test_2024_25["result"].value_counts())

print("\n2024/25 Duplicate Matches:")
print(
    test_2024_25.duplicated(
        subset=["season", "date", "home_team", "away_team"]
    ).sum()
)

print("\n")


# PROCESS A COMPLETE SEASON
def process_season(file):

    # Display the season currently being processed
    print(f"\nProcessing season: {file.name}")

    # Read the raw CSV file
    raw_df = pd.read_csv(file)

    # Clean the raw dataset
    clean_df = clean_season_data(raw_df)

    # Get the season name
    season = clean_df["season"].iloc[0]

    # Check for missing values
    missing_values = clean_df.isnull().sum().sum()

    # Check for duplicate matches
    duplicate_matches = clean_df.duplicated(
        subset=["season", "date", "home_team", "away_team"]
    ).sum()

    # Recalculate the result directly from the match goals
    calculated_result = clean_df.apply(
        lambda row: (
            "H" if row["home_goals"] > row["away_goals"]
            else "A" if row["home_goals"] < row["away_goals"]
            else "D"
        ),
        axis=1
    )

    # Count result mismatches
    result_mismatches = (
        clean_df["result"] != calculated_result
    ).sum()

    # Display validation results
    print(f"Season: {season}")
    print(f"Rows: {len(clean_df)}")
    print(f"Missing values: {missing_values}")
    print(f"Duplicate matches: {duplicate_matches}")
    print(f"Result mismatches: {result_mismatches}")

    # Display result distribution
    print("\nResult distribution:")
    print(clean_df["result"].value_counts())

    # Return the cleaned season dataset
    return clean_df


print("\n")

# DATA FOLDER SETUP
# Location of the original/raw datasets
RAW_DIR = Path("data/raw")

# Location where cleaned/processed datasets will be saved
PROCESSED_DIR = Path("data/processed")

# Create the processed folder if it does not already exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# FIND RAW DATASET FILES
# Find every CSV file inside the raw data folder
raw_files = list(RAW_DIR.glob("*.csv"))

# Display the number of CSV files found
print("Raw CSV files found:", len(raw_files))

# Display the name/path of each raw CSV file
for file in raw_files:
    print(file)


print("\n")

# PROCESS ALL RAW SEASONS

# Create an empty list to store the processed seasons
cleaned_seasons = []

# Process every raw CSV file
for file in raw_files:

    # Process the complete season
    clean_df = process_season(file)

    # Store the processed season
    cleaned_seasons.append(clean_df)


   
# CHECK RESULT DISTRIBUTION BY SEASON
# Go through each cleaned season
for clean_df in cleaned_seasons:

    # Get the season name
    season = clean_df["season"].iloc[0]

    # Count the number of Home Wins, Draws, and Away Wins
    result_counts = clean_df["result"].value_counts()

    # Calculate the percentage of each result
    result_percentages = clean_df["result"].value_counts(normalize=True) * 100

    print(f"\n--- {season} Result Distribution ---")

    print("\nNumber of matches:")
    print(result_counts)

    print("\nPercentage of matches:")
    print(result_percentages.round(2))



# VALIDATE RESULTS AGAINST MATCH GOALS
# Go through each cleaned season
for clean_df in cleaned_seasons:

    # Recalculate the result directly from the recorded goals
    calculated_result = clean_df.apply(
        lambda row: (
            "H" if row["home_goals"] > row["away_goals"]
            else "A" if row["home_goals"] < row["away_goals"]
            else "D"
        ),
        axis=1
    )

    # Compare the existing result with the recalculated result
    mismatches = (clean_df["result"] != calculated_result).sum()

    # Get the season name
    season = clean_df["season"].iloc[0]

    print(f"\n--- {season} Result Validation ---")
    print("Result mismatches:", mismatches)



# CHECK DATA QUALITY BY SEASON
# Go through each cleaned season
for clean_df in cleaned_seasons:

    # Get the season name
    season = clean_df["season"].iloc[0]

    # Count all missing values in the dataset
    missing_values = clean_df.isnull().sum().sum()

    # Check for duplicate matches
    duplicate_matches = clean_df.duplicated(
        subset=["season", "date", "home_team", "away_team"]
    ).sum()

    print(f"\n--- {season} Data Quality Check ---")

    print("Missing values:", missing_values)
    print("Duplicate matches:", duplicate_matches)



# COMBINE ALL CLEANED SEASONS
# Combine the cleaned datasets into one DataFrame
combined_df = pd.concat(cleaned_seasons, ignore_index=True)


# STANDARDIZE SEASON NAMES
# Convert season names such as 2010/11 to 2010-11
combined_df["season"] = combined_df["season"].str.replace(
    "/", "-", regex=False
)


# Display the shape of the combined dataset
print("\n--- Combined Dataset ---")
print("Shape:", combined_df.shape)

# Display the number of matches from each season
print("\nMatches by season:")
print(combined_df["season"].value_counts())



# VALIDATE COMBINED DATASET
# Check for missing values in the combined dataset
print("\nMissing values:")
print(combined_df.isnull().sum())

# Check for duplicate matches across all seasons
duplicate_matches = combined_df.duplicated(
    subset=["season", "date", "home_team", "away_team"]
).sum()

print("\nDuplicate matches:", duplicate_matches)

# Check the overall result distribution
print("\nOverall Result Distribution:")
print(combined_df["result"].value_counts())

# Check the overall result percentages
result_percentages = combined_df["result"].value_counts(normalize=True) * 100

print("\nOverall Result Percentages:")
print(result_percentages.round(2))


print("\n")

# SAVE COMBINED PROCESSED DATASET
# Define the location and filename for the processed dataset
output_file = PROCESSED_DIR / "champions_league_all_seasons_clean.csv"

# Save the combined dataset
combined_df.to_csv(output_file, index=False)

# Confirm that the file was saved successfully
print("\nProcessed dataset saved successfully.")
print("Saved to:", output_file)


print("\n")

# Expected Seasons
expected_seasons = [
    "2010-11", "2011-12", "2012-13", "2013-14",
    "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26"
]


# Master Dataset Validation
print("\n===== MASTER DATASET VALIDATION =====")


# Total Matches
print("Total matches:", len(combined_df))


# Total Seasons
print("Total seasons:", combined_df["season"].nunique())


# Seasons Found
print("\nSeasons found:")
print(sorted(combined_df["season"].unique()))


# Missing Values
print("\nMissing values:")
print(combined_df.isnull().sum())


# Duplicate Matches
print("\nDuplicate matches:")
print(
    combined_df.duplicated(
        subset=["season", "date", "home_team", "away_team"]
    ).sum()
)


# Result Distribution
print("\nResult distribution:")
print(combined_df["result"].value_counts())


# Validation Checks
assert combined_df["season"].nunique() == 16
assert len(combined_df) == 2122
assert sorted(combined_df["season"].unique()) == expected_seasons

print("\n")

# Save Master Clean Dataset
output_file = PROCESSED_DIR / "champions_league_all_seasons_clean.csv"

combined_df.to_csv(output_file, index=False)

print("\nMaster dataset saved successfully.")
print("Saved to:", output_file)



print("\n")

# FEATURE ENGINEERING V1
# Create features using only information available before each match

# LOAD MASTER CLEAN DATASET
feature_df = combined_df.copy()


# CONVERT DATE COLUMN TO DATETIME
feature_df["date"] = pd.to_datetime(feature_df["date"])


# SORT MATCHES CHRONOLOGICALLY
feature_df = feature_df.sort_values(
    ["date", "home_team", "away_team"]
).reset_index(drop=True)


# FUNCTION TO CREATE EMPTY TEAM STATISTICS
def create_team_stats():
    return {
        "matches": 0,
        "goals_for": 0,
        "goals_against": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "home_matches": 0,
        "home_wins": 0,
        "away_matches": 0,
        "away_wins": 0,
        "recent_points": deque(maxlen=5),
        "last_match_date": None
    }


# STORE TEAM HISTORY
team_stats = {}


# FUNCTION TO GET CURRENT TEAM HISTORY
def get_team_stats(team):

    if team in team_stats:
        return team_stats[team]

    return create_team_stats()


# CREATE LIST FOR FEATURE ROWS
feature_rows = []


# PROCESS MATCHES BY DATE
for match_date, day_matches in feature_df.groupby("date", sort=True):


    # CREATE PRE-MATCH FEATURES
    for _, row in day_matches.iterrows():

        home_team = row["home_team"]
        away_team = row["away_team"]

        home_stats = get_team_stats(home_team)
        away_stats = get_team_stats(away_team)


        # HOME TEAM MATCH COUNT
        home_matches = home_stats["matches"]


        # AWAY TEAM MATCH COUNT
        away_matches = away_stats["matches"]


        # HOME TEAM AVERAGE GOALS SCORED
        home_avg_goals_for = (
            home_stats["goals_for"] / home_matches
            if home_matches > 0
            else 0
        )


        # HOME TEAM AVERAGE GOALS CONCEDED
        home_avg_goals_against = (
            home_stats["goals_against"] / home_matches
            if home_matches > 0
            else 0
        )


        # AWAY TEAM AVERAGE GOALS SCORED
        away_avg_goals_for = (
            away_stats["goals_for"] / away_matches
            if away_matches > 0
            else 0
        )


        # AWAY TEAM AVERAGE GOALS CONCEDED
        away_avg_goals_against = (
            away_stats["goals_against"] / away_matches
            if away_matches > 0
            else 0
        )


        # HOME TEAM WIN RATE
        home_win_rate = (
            home_stats["wins"] / home_matches
            if home_matches > 0
            else 0
        )


        # AWAY TEAM WIN RATE
        away_win_rate = (
            away_stats["wins"] / away_matches
            if away_matches > 0
            else 0
        )


        # HOME TEAM HOME WIN RATE
        home_home_win_rate = (
            home_stats["home_wins"] / home_stats["home_matches"]
            if home_stats["home_matches"] > 0
            else 0
        )


        # AWAY TEAM AWAY WIN RATE
        away_away_win_rate = (
            away_stats["away_wins"] / away_stats["away_matches"]
            if away_stats["away_matches"] > 0
            else 0
        )


        # HOME TEAM RECENT FORM
        home_recent_points = sum(home_stats["recent_points"])


        # AWAY TEAM RECENT FORM
        away_recent_points = sum(away_stats["recent_points"])


        # DAYS SINCE HOME TEAM LAST MATCH
        if home_stats["last_match_date"] is not None:

            home_days_since_match = (
                match_date - home_stats["last_match_date"]
            ).days

        else:

            home_days_since_match = -1


        # DAYS SINCE AWAY TEAM LAST MATCH
        if away_stats["last_match_date"] is not None:

            away_days_since_match = (
                match_date - away_stats["last_match_date"]
            ).days

        else:

            away_days_since_match = -1


        # STORE PRE-MATCH FEATURES
        feature_rows.append({

            "season": row["season"],
            "date": row["date"],
            "round": row["round"],

            "home_team": home_team,
            "away_team": away_team,

            "home_matches_before": home_matches,
            "away_matches_before": away_matches,

            "home_avg_goals_for": home_avg_goals_for,
            "home_avg_goals_against": home_avg_goals_against,

            "away_avg_goals_for": away_avg_goals_for,
            "away_avg_goals_against": away_avg_goals_against,

            "home_win_rate": home_win_rate,
            "away_win_rate": away_win_rate,

            "home_home_win_rate": home_home_win_rate,
            "away_away_win_rate": away_away_win_rate,

            "home_recent_points_5": home_recent_points,
            "away_recent_points_5": away_recent_points,

            "home_days_since_match": home_days_since_match,
            "away_days_since_match": away_days_since_match,

            # TARGET
            "result": row["result"]
        })


    # UPDATE TEAM HISTORY AFTER THE MATCHES
    for _, row in day_matches.iterrows():

        home_team = row["home_team"]
        away_team = row["away_team"]

        home_goals = row["home_goals"]
        away_goals = row["away_goals"]

        result = row["result"]


        # CREATE TEAM RECORDS IF NEEDED
        if home_team not in team_stats:
            team_stats[home_team] = create_team_stats()

        if away_team not in team_stats:
            team_stats[away_team] = create_team_stats()


        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]


        # UPDATE GENERAL MATCH STATISTICS
        home_stats["matches"] += 1
        away_stats["matches"] += 1

        home_stats["goals_for"] += home_goals
        home_stats["goals_against"] += away_goals

        away_stats["goals_for"] += away_goals
        away_stats["goals_against"] += home_goals


        # UPDATE HOME/AWAY MATCH COUNTS
        home_stats["home_matches"] += 1
        away_stats["away_matches"] += 1


        # UPDATE RESULT INFORMATION
        if result == "H":

            home_stats["wins"] += 1
            away_stats["losses"] += 1

            home_stats["home_wins"] += 1

            home_stats["recent_points"].append(3)
            away_stats["recent_points"].append(0)


        elif result == "A":

            home_stats["losses"] += 1
            away_stats["wins"] += 1

            away_stats["away_wins"] += 1

            home_stats["recent_points"].append(0)
            away_stats["recent_points"].append(3)


        else:

            home_stats["draws"] += 1
            away_stats["draws"] += 1

            home_stats["recent_points"].append(1)
            away_stats["recent_points"].append(1)


        # UPDATE LAST MATCH DATE
        home_stats["last_match_date"] = match_date
        away_stats["last_match_date"] = match_date


# CREATE FINAL FEATURE DATAFRAME
features_df = pd.DataFrame(feature_rows)


# CHECK FEATURE DATASET SHAPE
print("\n===== FEATURE DATASET =====")

print("Shape:", features_df.shape)


# CHECK FEATURE COLUMNS
print("\nFeature columns:")

print(features_df.columns.tolist())


# PREVIEW FEATURES
print("\nFirst 10 rows:")

print(features_df.head(10))


# CHECK MISSING VALUES
print("\nMissing values:")

print(features_df.isnull().sum())


# CHECK TARGET DISTRIBUTION
print("\nTarget distribution:")

print(features_df["result"].value_counts())


# SAVE FEATURE DATASET
features_output = PROCESSED_DIR / "champions_league_features_v1.csv"

features_df.to_csv(
    features_output,
    index=False
)

print("\nFeature dataset saved successfully.")
print("Saved to:", features_output)




# INSPECT EARLY-MATCH FEATURES
print("\n===== EARLY-MATCH FEATURE INSPECTION =====")

# Check how many matches each team had played before the current match
print("\nHome matches played before current match:")
print(features_df["home_matches_before"].describe())

print("\nAway matches played before current match:")
print(features_df["away_matches_before"].describe())


# CHECK MATCHES WITH NO PRIOR MATCH HISTORY
home_no_history = (
    features_df["home_matches_before"] == 0
).sum()

away_no_history = (
    features_df["away_matches_before"] == 0
).sum()

print("\nMatches where home team had no prior matches:")
print(home_no_history)

print("\nMatches where away team had no prior matches:")
print(away_no_history)


# SHOW EARLY-MATCH ROWS
early_matches = features_df[
    (features_df["home_matches_before"] <= 2) |
    (features_df["away_matches_before"] <= 2)
]

print("\n===== EARLY-MATCH ROWS =====")
print("Number of early-history matches:", len(early_matches))

print(
    early_matches[
        [
            "season",
            "date",
            "home_team",
            "away_team",
            "home_matches_before",
            "away_matches_before",
            "home_avg_goals_for",
            "away_avg_goals_for",
            "home_win_rate",
            "away_win_rate",
            "home_recent_points_5",
            "away_recent_points_5",
            "result"
        ]
    ].head(20)
)


# CHECK DAYS SINCE LAST MATCH
print("\n===== DAYS SINCE LAST MATCH =====")

print(
    "\nHome days since previous match:"
)
print(
    features_df["home_days_since_match"].describe()
)

print(
    "\nAway days since previous match:"
)
print(
    features_df["away_days_since_match"].describe()
)

print(
    "\nHome teams with no previous match:"
)
print(
    (features_df["home_days_since_match"] == -1).sum()
)

print(
    "\nAway teams with no previous match:"
)
print(
    (features_df["away_days_since_match"] == -1).sum()
)


# CHECK FEATURE RANGES
print("\n===== FEATURE RANGE CHECK =====")

feature_columns = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "home_win_rate",
    "away_win_rate",
    "home_home_win_rate",
    "away_away_win_rate",
    "home_recent_points_5",
    "away_recent_points_5"
]

print(
    features_df[feature_columns].describe().T
)


# CHECK WIN RATE VALIDITY
print("\n===== WIN RATE VALIDATION =====")

print(
    "Home win rate below 0:",
    (features_df["home_win_rate"] < 0).sum()
)

print(
    "Home win rate above 1:",
    (features_df["home_win_rate"] > 1).sum()
)

print(
    "Away win rate below 0:",
    (features_df["away_win_rate"] < 0).sum()
)

print(
    "Away win rate above 1:",
    (features_df["away_win_rate"] > 1).sum()
)


# CHECK RECENT POINTS
print("\n===== RECENT POINTS VALIDATION =====")

print(
    "Home recent points range:",
    features_df["home_recent_points_5"].min(),
    "to",
    features_df["home_recent_points_5"].max()
)

print(
    "Away recent points range:",
    features_df["away_recent_points_5"].min(),
    "to",
    features_df["away_recent_points_5"].max()
)


# CHECK FOR COMPLETELY UNINFORMATIVE ROWS
uninformative_rows = features_df[
    (features_df["home_matches_before"] == 0) &
    (features_df["away_matches_before"] == 0)
]

print("\n===== BOTH TEAMS WITH NO PRIOR HISTORY =====")
print(
    "Number of matches:",
    len(uninformative_rows)
)

print(
    uninformative_rows[
        [
            "season",
            "date",
            "home_team",
            "away_team",
            "result"
        ]
    ].head(20)
)



print("\n")


# FEATURE ENGINEERING V1.1
# IMPROVE DAYS SINCE LAST MATCH
features_v1_1 = features_df.copy()


# ADD PREVIOUS MATCH INDICATORS
features_v1_1["home_has_previous_match"] = (
    features_v1_1["home_days_since_match"] >= 0
).astype(int)

features_v1_1["away_has_previous_match"] = (
    features_v1_1["away_days_since_match"] >= 0
).astype(int)


# CREATE CAPPED DAYS SINCE LAST MATCH
features_v1_1["home_days_since_match_capped"] = (
    features_v1_1["home_days_since_match"]
    .clip(lower=0, upper=365)
)

features_v1_1["away_days_since_match_capped"] = (
    features_v1_1["away_days_since_match"]
    .clip(lower=0, upper=365)
)


# DISPLAY NEW FEATURES
print("\n===== FEATURE DATASET V1.1 =====")

print(
    "\nNew columns:"
)

print(
    [
        "home_has_previous_match",
        "away_has_previous_match",
        "home_days_since_match_capped",
        "away_days_since_match_capped"
    ]
)


# CHECK PREVIOUS MATCH INDICATORS
print("\n===== PREVIOUS MATCH INDICATORS =====")

print(
    "\nHome team previous-match indicator:"
)

print(
    features_v1_1["home_has_previous_match"].value_counts()
)

print(
    "\nAway team previous-match indicator:"
)

print(
    features_v1_1["away_has_previous_match"].value_counts()
)


# CHECK CAPPED DAYS
print("\n===== CAPPED DAYS CHECK =====")

print(
    "\nHome capped days:"
)

print(
    features_v1_1["home_days_since_match_capped"].describe()
)

print(
    "\nAway capped days:"
)

print(
    features_v1_1["away_days_since_match_capped"].describe()
)


# CHECK MAXIMUM VALUES
print("\nMaximum home capped days:")

print(
    features_v1_1["home_days_since_match_capped"].max()
)

print("\nMaximum away capped days:")

print(
    features_v1_1["away_days_since_match_capped"].max()
)


# CHECK NEW EARLY-MATCH ROWS
print("\n===== EARLY-MATCH V1.1 EXAMPLE =====")

early_v1_1 = features_v1_1[
    (
        features_v1_1["home_has_previous_match"] == 0
    )
    |
    (
        features_v1_1["away_has_previous_match"] == 0
    )
]

print(
    early_v1_1[
        [
            "season",
            "date",
            "home_team",
            "away_team",
            "home_matches_before",
            "away_matches_before",
            "home_has_previous_match",
            "away_has_previous_match",
            "home_days_since_match",
            "away_days_since_match",
            "home_days_since_match_capped",
            "away_days_since_match_capped",
            "result"
        ]
    ].head(20)
)


# CHECK MISSING VALUES
print("\n===== V1.1 MISSING VALUES =====")

print(
    features_v1_1.isnull().sum()
)


# SAVE V1.1 DATASET
features_v1_1_output = (
    PROCESSED_DIR /
    "champions_league_features_v1_1.csv"
)

features_v1_1.to_csv(
    features_v1_1_output,
    index=False
)

print("\nFeature dataset V1.1 saved successfully.")

print(
    "Saved to:",
    features_v1_1_output
)





# FINAL FEATURE SANITY CHECK
print("\n===== FINAL FEATURE SANITY CHECK =====")

print("\nDataset shape:")
print(features_v1_1.shape)

print("\nData types:")
print(features_v1_1.dtypes)

print("\nDuplicate matches:")
print(
    features_v1_1.duplicated(
        subset=["season", "date", "home_team", "away_team"]
    ).sum()
)

print("\nMissing values:")
print(features_v1_1.isnull().sum())

print("\nTarget distribution:")
print(features_v1_1["result"].value_counts())

print("\nTarget percentages:")
print(
    (
        features_v1_1["result"]
        .value_counts(normalize=True) * 100
    ).round(2)
)


# CHECK THAT FEATURES COME BEFORE THE TARGET
print("\n===== FEATURE ORDER CHECK =====")

print(
    features_v1_1.columns.tolist()
)


# DEFINE MODEL FEATURES
model_features = [
    "home_matches_before",
    "away_matches_before",

    "home_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_for",
    "away_avg_goals_against",

    "home_win_rate",
    "away_win_rate",

    "home_home_win_rate",
    "away_away_win_rate",

    "home_recent_points_5",
    "away_recent_points_5",

    "home_days_since_match_capped",
    "away_days_since_match_capped",

    "home_has_previous_match",
    "away_has_previous_match"
]


X = features_v1_1[model_features]
y = features_v1_1["result"]


print("\nNumber of model features:")
print(len(model_features))

print("\nModel features:")
print(model_features)


# TIME-AWARE TRAIN / TEST SPLIT
print("\n===== TIME-AWARE TRAIN / TEST SPLIT =====")

# Sort the dataset chronologically
model_df = features_v1_1.sort_values(
    "date"
).reset_index(drop=True)


# FIND AN APPROXIMATE 80% SPLIT POINT
split_index = int(len(model_df) * 0.80)

# Get the date at the initial split point
cutoff_date = model_df.loc[
    split_index,
    "date"
]

print("\nInitial cutoff date:")
print(cutoff_date)


# KEEP THE ENTIRE CUTOFF DATE TOGETHER
train_df = model_df[
    model_df["date"] < cutoff_date
].copy()

test_df = model_df[
    model_df["date"] >= cutoff_date
].copy()


# DISPLAY TRAIN / TEST INFORMATION
print("\nTraining rows:")
print(len(train_df))

print("\nTesting rows:")
print(len(test_df))


print("\nTraining date range:")
print(
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print("\nTesting date range:")
print(
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)


print("\nTraining seasons:")
print(
    sorted(train_df["season"].unique())
)

print("\nTesting seasons:")
print(
    sorted(test_df["season"].unique())
)


# CREATE X AND Y FOR TRAINING AND TESTING
X_train = train_df[model_features]
y_train = train_df["result"]

X_test = test_df[model_features]
y_test = test_df["result"]


print("\n===== FINAL TRAIN / TEST SHAPES =====")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# TEMPORAL LEAKAGE CHECK
print("\n===== TEMPORAL LEAKAGE CHECK =====")

last_train_date = train_df["date"].max()
first_test_date = test_df["date"].min()

print(
    "Last training date:",
    last_train_date
)

print(
    "First testing date:",
    first_test_date
)


if last_train_date < first_test_date:
    print("Temporal split verified: PASS")
else:
    print("Temporal split verified: FAIL")


# CHECK FOR DATE OVERLAP
overlapping_dates = set(
    train_df["date"]
).intersection(
    set(test_df["date"])
)

print("\nOverlapping dates:")
print(len(overlapping_dates))

if len(overlapping_dates) == 0:
    print("Date overlap check: PASS")
else:
    print("Date overlap check: FAIL")


print("\n")



# CREATE THE MODEL PIPELINE
logistic_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)


# TRAIN THE MODEL
print("\n===== TRAINING LOGISTIC REGRESSION =====")

logistic_model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")


# MAKE CLASS PREDICTIONS
y_pred = logistic_model.predict(
    X_test
)


# GET PREDICTED PROBABILITIES
y_prob = logistic_model.predict_proba(
    X_test
)

print("\nProbability array shape:")
print(y_prob.shape)


# SHOW MODEL CLASSES
model_classes = logistic_model.named_steps[
    "model"
].classes_

print("\nModel classes:")
print(model_classes)


# ACCURACY
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy:")
print(round(accuracy, 4))


# LOG LOSS
logloss = log_loss(
    y_test,
    y_prob,
    labels=model_classes
)

print("\nLog Loss:")
print(round(logloss, 4))


# CONFUSION MATRIX
print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model_classes
)

print(cm)


# CLASSIFICATION REPORT
print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        labels=model_classes
    )
)


# CHECK FIRST PREDICTIONS
print("\n===== FIRST 10 PREDICTIONS =====")

prediction_preview = test_df[
    [
        "season",
        "date",
        "home_team",
        "away_team",
        "result"
    ]
].copy()

prediction_preview["predicted_result"] = y_pred

prediction_preview["home_win_probability"] = (
    y_prob[:, list(model_classes).index("H")]
)

prediction_preview["draw_probability"] = (
    y_prob[:, list(model_classes).index("D")]
)

prediction_preview["away_win_probability"] = (
    y_prob[:, list(model_classes).index("A")]
)


print(
    prediction_preview.head(10)
)


# MULTICLASS BRIER SCORE
print("\n===== MULTICLASS BRIER SCORE =====")

class_to_index = {
    class_name: index
    for index, class_name in enumerate(model_classes)
}

y_true_encoded = y_test.map(
    class_to_index
).to_numpy()

brier_score = (
    (
        (y_prob -
         (
             y_true_encoded[:, None]
             == range(len(model_classes))
         )
         .astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()

print(
    "Brier Score:",
    round(brier_score, 4)
)





# BASELINE MODEL DIAGNOSIS
print("\n===== BASELINE MODEL DIAGNOSIS =====")


# PREDICTED CLASS DISTRIBUTION
predicted_counts = (
    pd.Series(y_pred)
    .value_counts()
    .reindex(model_classes, fill_value=0)
)

predicted_percentages = (
    predicted_counts /
    len(y_pred)
) * 100

print("\nPredicted class counts:")
print(predicted_counts)

print("\nPredicted class percentages:")
print(predicted_percentages.round(2))


# ACTUAL TEST CLASS DISTRIBUTION
actual_counts = (
    y_test
    .value_counts()
    .reindex(model_classes, fill_value=0)
)

actual_percentages = (
    actual_counts /
    len(y_test)
) * 100

print("\nActual test class counts:")
print(actual_counts)

print("\nActual test class percentages:")
print(actual_percentages.round(2))


# AVERAGE PREDICTED PROBABILITIES
average_probabilities = pd.Series(
    y_prob.mean(axis=0),
    index=model_classes
)

print("\nAverage predicted probabilities:")
print(
    average_probabilities.round(4)
)

print("\nAverage predicted probabilities (%):")
print(
    (average_probabilities * 100).round(2)
)


# COMPARE PREDICTED PROBABILITY WITH ACTUAL FREQUENCY
probability_comparison = pd.DataFrame({
    "average_predicted_probability": average_probabilities,
    "actual_frequency": (
        y_test.value_counts(normalize=True)
        .reindex(model_classes)
    )
})

print("\n===== PROBABILITY VS ACTUAL FREQUENCY =====")

print(
    probability_comparison.round(4)
)


# CHECK DRAW PROBABILITIES
draw_index = list(model_classes).index("D")

draw_probabilities = y_prob[:, draw_index]

print("\n===== DRAW PROBABILITY CHECK =====")

print(
    "Average draw probability:",
    round(draw_probabilities.mean(), 4)
)

print(
    "Maximum draw probability:",
    round(draw_probabilities.max(), 4)
)

print(
    "Minimum draw probability:",
    round(draw_probabilities.min(), 4)
)


# SHOW HIGHEST DRAW PROBABILITY MATCHES
draw_check = test_df[
    [
        "season",
        "date",
        "home_team",
        "away_team",
        "result"
    ]
].copy()

draw_check["draw_probability"] = draw_probabilities

print("\n===== HIGHEST DRAW PROBABILITY MATCHES =====")

print(
    draw_check
    .sort_values(
        "draw_probability",
        ascending=False
    )
    .head(10)
)


# MAJORITY-CLASS BASELINE
print("\n===== MAJORITY / HISTORICAL BASELINE =====")

train_class_probabilities = (
    y_train
    .value_counts(normalize=True)
    .reindex(model_classes)
    .fillna(0)
)

print("\nTraining-set class probabilities:")
print(
    train_class_probabilities.round(4)
)


# Give every test match the historical training probabilities
baseline_probabilities = np.tile(
    train_class_probabilities.to_numpy(),
    (len(y_test), 1)
)


baseline_log_loss = log_loss(
    y_test,
    baseline_probabilities,
    labels=model_classes
)

print("\nHistorical baseline Log Loss:")
print(
    round(baseline_log_loss, 4)
)


# MODEL VS BASELINE
print("\n===== MODEL VS BASELINE =====")

print(
    "Logistic Regression Log Loss:",
    round(logloss, 4)
)

print(
    "Historical Baseline Log Loss:",
    round(baseline_log_loss, 4)
)

if logloss < baseline_log_loss:
    print(
        "Logistic Regression beats the historical probability baseline."
    )
else:
    print(
        "Logistic Regression does not beat the historical probability baseline."
    )


# NORMALIZED MULTICLASS BRIER SCORE
normalized_brier_score = brier_score / len(model_classes)

print("\nNormalized multiclass Brier Score:")
print(
    round(normalized_brier_score, 4)
)



print("\n BASELINE MODEL 2 Random Forest")

# BASELINE MODEL 2
# RANDOM FOREST
# CREATE RANDOM FOREST MODEL
random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# TRAIN RANDOM FOREST
print("\n===== TRAINING RANDOM FOREST =====")

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed successfully.")


# MAKE PREDICTIONS
rf_y_pred = random_forest_model.predict(
    X_test
)

rf_y_prob = random_forest_model.predict_proba(
    X_test
)


# GET MODEL CLASSES
rf_classes = random_forest_model.classes_

print("\nRandom Forest classes:")
print(rf_classes)


# ACCURACY
rf_accuracy = accuracy_score(
    y_test,
    rf_y_pred
)

print("\n===== RANDOM FOREST PERFORMANCE =====")

print(
    "Accuracy:",
    round(rf_accuracy, 4)
)


# LOG LOSS
rf_logloss = log_loss(
    y_test,
    rf_y_prob,
    labels=rf_classes
)

print(
    "Log Loss:",
    round(rf_logloss, 4)
)


# CONFUSION MATRIX
rf_cm = confusion_matrix(
    y_test,
    rf_y_pred,
    labels=rf_classes
)

print("\n===== RANDOM FOREST CONFUSION MATRIX =====")

print(rf_cm)


# CLASSIFICATION REPORT
print("\n===== RANDOM FOREST CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        rf_y_pred,
        labels=rf_classes
    )
)


# MULTICLASS BRIER SCORE
rf_class_to_index = {
    class_name: index
    for index, class_name in enumerate(rf_classes)
}

rf_y_true_encoded = y_test.map(
    rf_class_to_index
).to_numpy()

rf_brier_score = (
    (
        (
            rf_y_prob -
            (
                rf_y_true_encoded[:, None]
                == range(len(rf_classes))
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()

print(
    "\nRandom Forest Brier Score:",
    round(rf_brier_score, 4)
)

print(
    "Random Forest Normalized Brier Score:",
    round(
        rf_brier_score / len(rf_classes),
        4
    )
)


# COMPARE THE TWO MODELS
print("\n===== MODEL COMPARISON =====")

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy
    ],
    "Log Loss": [
        logloss,
        rf_logloss
    ],
    "Brier Score": [
        brier_score,
        rf_brier_score
    ],
    "Normalized Brier": [
        normalized_brier_score,
        rf_brier_score / len(rf_classes)
    ]
})

print(
    model_comparison.round(4)
)


# RANDOM FOREST PROBABILITY PREVIEW
rf_prediction_preview = test_df[
    [
        "season",
        "date",
        "home_team",
        "away_team",
        "result"
    ]
].copy()

rf_prediction_preview["predicted_result"] = rf_y_pred

rf_prediction_preview["away_win_probability"] = (
    rf_y_prob[
        :,
        list(rf_classes).index("A")
    ]
)

rf_prediction_preview["draw_probability"] = (
    rf_y_prob[
        :,
        list(rf_classes).index("D")
    ]
)

rf_prediction_preview["home_win_probability"] = (
    rf_y_prob[
        :,
        list(rf_classes).index("H")
    ]
)

print("\n===== RANDOM FOREST FIRST 10 PREDICTIONS =====")

print(
    rf_prediction_preview.head(10)
)




print("\n # BASELINE MODEL 3 CATBOOST")

# BASELINE MODEL 3
# CATBOOST
catboost_model = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.03,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False
)


# TRAIN CATBOOST
print("\n===== TRAINING CATBOOST =====")

catboost_model.fit(
    X_train,
    y_train
)

print(
    "CatBoost training completed successfully."
)


# MAKE PREDICTIONS
cb_y_pred = catboost_model.predict(
    X_test
).ravel()

cb_y_prob = catboost_model.predict_proba(
    X_test
)


# GET MODEL CLASSES
cb_classes = catboost_model.classes_

print("\nCatBoost classes:")
print(cb_classes)


# ACCURACY
cb_accuracy = accuracy_score(
    y_test,
    cb_y_pred
)

print("\n===== CATBOOST PERFORMANCE =====")

print(
    "Accuracy:",
    round(cb_accuracy, 4)
)


# LOG LOSS
cb_logloss = log_loss(
    y_test,
    cb_y_prob,
    labels=cb_classes
)

print(
    "Log Loss:",
    round(cb_logloss, 4)
)


# CONFUSION MATRIX
cb_cm = confusion_matrix(
    y_test,
    cb_y_pred,
    labels=cb_classes
)

print("\n===== CATBOOST CONFUSION MATRIX =====")

print(cb_cm)


# CLASSIFICATION REPORT
print("\n===== CATBOOST CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        cb_y_pred,
        labels=cb_classes
    )
)


# MULTICLASS BRIER SCORE
cb_class_to_index = {
    class_name: index
    for index, class_name in enumerate(cb_classes)
}

cb_y_true_encoded = y_test.map(
    cb_class_to_index
).to_numpy()

cb_brier_score = (
    (
        (
            cb_y_prob -
            (
                cb_y_true_encoded[:, None]
                == range(len(cb_classes))
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()

cb_normalized_brier = (
    cb_brier_score /
    len(cb_classes)
)

print(
    "\nCatBoost Brier Score:",
    round(cb_brier_score, 4)
)

print(
    "CatBoost Normalized Brier Score:",
    round(cb_normalized_brier, 4)
)


# COMPARE ALL THREE MODELS
print("\n===== ALL MODEL COMPARISON =====")

all_model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "CatBoost"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy,
        cb_accuracy
    ],
    "Log Loss": [
        logloss,
        rf_logloss,
        cb_logloss
    ],
    "Brier Score": [
        brier_score,
        rf_brier_score,
        cb_brier_score
    ],
    "Normalized Brier": [
        normalized_brier_score,
        rf_brier_score / len(rf_classes),
        cb_normalized_brier
    ]
})

print(
    all_model_comparison.round(4)
)


# CATBOOST FIRST 10 PROBABILITIES
cb_prediction_preview = test_df[
    [
        "season",
        "date",
        "home_team",
        "away_team",
        "result"
    ]
].copy()

cb_prediction_preview["predicted_result"] = (
    cb_y_pred
)

cb_prediction_preview["away_win_probability"] = (
    cb_y_prob[
        :,
        list(cb_classes).index("A")
    ]
)

cb_prediction_preview["draw_probability"] = (
    cb_y_prob[
        :,
        list(cb_classes).index("D")
    ]
)

cb_prediction_preview["home_win_probability"] = (
    cb_y_prob[
        :,
        list(cb_classes).index("H")
    ]
)

print(
    "\n===== CATBOOST FIRST 10 PREDICTIONS ====="
)

print(
    cb_prediction_preview.head(10)
)



print("\n  BASELINE MODEL 4 XGBOOST")


# BASELINE MODEL 4
# XGBOOST
# ENCODE TARGET CLASSES FOR XGBOOST
xgb_label_encoder = LabelEncoder()

y_train_xgb = xgb_label_encoder.fit_transform(
    y_train
)

y_test_xgb = xgb_label_encoder.transform(
    y_test
)

print("\n===== XGBOOST CLASS ENCODING =====")

print(
    "Original classes:",
    xgb_label_encoder.classes_
)

print(
    "Encoded classes:",
    xgb_label_encoder.transform(
        xgb_label_encoder.classes_
    )
)


# CREATE XGBOOST MODEL
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


# TRAIN XGBOOST
print("\n===== TRAINING XGBOOST =====")

xgb_model.fit(
    X_train,
    y_train_xgb
)

print(
    "XGBoost training completed successfully."
)


# MAKE ENCODED PREDICTIONS
xgb_y_pred_encoded = xgb_model.predict(
    X_test
)

xgb_y_prob = xgb_model.predict_proba(
    X_test
)


# CONVERT PREDICTIONS BACK TO H / D / A
xgb_y_pred = xgb_label_encoder.inverse_transform(
    xgb_y_pred_encoded.astype(int)
)

xgb_classes = xgb_label_encoder.classes_


print("\nXGBoost classes:")
print(xgb_classes)


# ACCURACY
xgb_accuracy = accuracy_score(
    y_test,
    xgb_y_pred
)

print("\n===== XGBOOST PERFORMANCE =====")

print(
    "Accuracy:",
    round(xgb_accuracy, 4)
)


# LOG LOSS
xgb_logloss = log_loss(
    y_test,
    xgb_y_prob,
    labels=xgb_classes
)

print(
    "Log Loss:",
    round(xgb_logloss, 4)
)


# CONFUSION MATRIX
xgb_cm = confusion_matrix(
    y_test,
    xgb_y_pred,
    labels=xgb_classes
)

print("\n===== XGBOOST CONFUSION MATRIX =====")

print(xgb_cm)


# CLASSIFICATION REPORT
print("\n===== XGBOOST CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        xgb_y_pred,
        labels=xgb_classes
    )
)


# BRIER SCORE
xgb_class_to_index = {
    class_name: index
    for index, class_name in enumerate(xgb_classes)
}

xgb_y_true_encoded = y_test.map(
    xgb_class_to_index
).to_numpy()

xgb_brier_score = (
    (
        (
            xgb_y_prob -
            (
                xgb_y_true_encoded[:, None]
                == range(len(xgb_classes))
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()

xgb_normalized_brier = (
    xgb_brier_score /
    len(xgb_classes)
)

print(
    "\nXGBoost Brier Score:",
    round(xgb_brier_score, 4)
)

print(
    "XGBoost Normalized Brier Score:",
    round(xgb_normalized_brier, 4)
)


# FOUR MODEL COMPARISON
print("\n===== FOUR MODEL COMPARISON =====")

all_model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "CatBoost",
        "XGBoost"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy,
        cb_accuracy,
        xgb_accuracy
    ],
    "Log Loss": [
        logloss,
        rf_logloss,
        cb_logloss,
        xgb_logloss
    ],
    "Brier Score": [
        brier_score,
        rf_brier_score,
        cb_brier_score,
        xgb_brier_score
    ],
    "Normalized Brier": [
        normalized_brier_score,
        rf_brier_score / len(rf_classes),
        cb_normalized_brier,
        xgb_normalized_brier
    ]
})

print(
    all_model_comparison.round(4)
)


print("\n  FEATURE ENGINEERING V2")

# FEATURE ENGINEERING V2
# RELATIVE TEAM STRENGTH FEATURES
features_v2 = features_v1_1.copy()


# GOAL SCORING EDGE
features_v2["home_goal_scoring_edge"] = (
    features_v2["home_avg_goals_for"]
    - features_v2["away_avg_goals_for"]
)


# DEFENSIVE EDGE
# Positive value means the home team has conceded fewer
# goals on average than the away team.

features_v2["home_defensive_edge"] = (
    features_v2["away_avg_goals_against"]
    - features_v2["home_avg_goals_against"]
)


# HOME ATTACK VS AWAY DEFENSE
features_v2["home_attack_vs_away_defense"] = (
    features_v2["home_avg_goals_for"]
    - features_v2["away_avg_goals_against"]
)


# AWAY ATTACK VS HOME DEFENSE
features_v2["away_attack_vs_home_defense"] = (
    features_v2["away_avg_goals_for"]
    - features_v2["home_avg_goals_against"]
)


# OVERALL WIN RATE EDGE
features_v2["win_rate_edge"] = (
    features_v2["home_win_rate"]
    - features_v2["away_win_rate"]
)


# VENUE WIN RATE EDGE
features_v2["venue_win_rate_edge"] = (
    features_v2["home_home_win_rate"]
    - features_v2["away_away_win_rate"]
)


# RECENT FORM EDGE
features_v2["recent_points_edge"] = (
    features_v2["home_recent_points_5"]
    - features_v2["away_recent_points_5"]
)


# EXPERIENCE EDGE
features_v2["experience_edge"] = (
    features_v2["home_matches_before"]
    - features_v2["away_matches_before"]
)


# REST / MATCH-RECENCY EDGE
features_v2["days_since_match_edge"] = (
    features_v2["home_days_since_match_capped"]
    - features_v2["away_days_since_match_capped"]
)


# DISPLAY NEW FEATURES
new_v2_features = [
    "home_goal_scoring_edge",
    "home_defensive_edge",
    "home_attack_vs_away_defense",
    "away_attack_vs_home_defense",
    "win_rate_edge",
    "venue_win_rate_edge",
    "recent_points_edge",
    "experience_edge",
    "days_since_match_edge"
]

print("\n===== FEATURE ENGINEERING V2 =====")

print("\nNew relative features:")

for feature in new_v2_features:
    print("-", feature)


# CHECK NEW FEATURE STATISTICS
print("\n===== V2 FEATURE STATISTICS =====")

print(
    features_v2[new_v2_features]
    .describe()
    .round(4)
)


# CHECK MISSING VALUES
print("\n===== V2 MISSING VALUES =====")

print(
    features_v2[new_v2_features]
    .isnull()
    .sum()
)


# CHECK EXTREME VALUES
print("\n===== V2 FEATURE EXTREMES =====")

for feature in new_v2_features:

    print(
        f"\n{feature}"
    )

    print(
        "Minimum:",
        round(features_v2[feature].min(), 4)
    )

    print(
        "Maximum:",
        round(features_v2[feature].max(), 4)
    )


# CREATE V2 MODEL FEATURES
model_features_v2 = model_features + new_v2_features


print("\n===== V2 MODEL FEATURES =====")

print(
    "Number of features:",
    len(model_features_v2)
)

print(
    model_features_v2
)


# CREATE V2 MODEL DATA
X_v2 = features_v2[model_features_v2]

y_v2 = features_v2["result"]


# SORT CHRONOLOGICALLY
model_df_v2 = features_v2.sort_values(
    "date"
).reset_index(drop=True)


# TIME-AWARE TRAIN / TEST SPLIT
split_index_v2 = int(
    len(model_df_v2) * 0.80
)

cutoff_date_v2 = model_df_v2.loc[
    split_index_v2,
    "date"
]

train_df_v2 = model_df_v2[
    model_df_v2["date"] < cutoff_date_v2
].copy()

test_df_v2 = model_df_v2[
    model_df_v2["date"] >= cutoff_date_v2
].copy()


# CREATE X / Y DATASETS
X_train_v2 = train_df_v2[
    model_features_v2
]

y_train_v2 = train_df_v2[
    "result"
]

X_test_v2 = test_df_v2[
    model_features_v2
]

y_test_v2 = test_df_v2[
    "result"
]


# FINAL V2 SPLIT CHECK
print("\n===== V2 TRAIN / TEST SPLIT =====")

print(
    "Training rows:",
    len(train_df_v2)
)

print(
    "Testing rows:",
    len(test_df_v2)
)

print(
    "Training date:",
    train_df_v2["date"].min(),
    "to",
    train_df_v2["date"].max()
)

print(
    "Testing date:",
    test_df_v2["date"].min(),
    "to",
    test_df_v2["date"].max()
)


# TEMPORAL LEAKAGE CHECK
v2_overlapping_dates = set(
    train_df_v2["date"]
).intersection(
    set(test_df_v2["date"])
)

print(
    "\nOverlapping dates:",
    len(v2_overlapping_dates)
)

if (
    train_df_v2["date"].max()
    < test_df_v2["date"].min()
    and
    len(v2_overlapping_dates) == 0
):
    print(
        "V2 temporal split verified: PASS"
    )
else:
    print(
        "V2 temporal split verified: FAIL"
    )


# SAVE FEATURE ENGINEERING V2
features_v2_output = (
    PROCESSED_DIR /
    "champions_league_features_v2.csv"
)

features_v2.to_csv(
    features_v2_output,
    index=False
)

print(
    "\nFeature dataset V2 saved successfully."
)

print(
    "Saved to:",
    features_v2_output
)




# V2 MODEL TESTING
# LOGISTIC REGRESSION + CATBOOST
# V2 LOGISTIC REGRESSION
print("\n===== V2 LOGISTIC REGRESSION =====")

logistic_model_v2 = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)

logistic_model_v2.fit(
    X_train_v2,
    y_train_v2
)

v2_lr_pred = logistic_model_v2.predict(
    X_test_v2
)

v2_lr_prob = logistic_model_v2.predict_proba(
    X_test_v2
)

v2_lr_classes = (
    logistic_model_v2
    .named_steps["model"]
    .classes_
)


# V2 LOGISTIC REGRESSION METRICS
v2_lr_accuracy = accuracy_score(
    y_test_v2,
    v2_lr_pred
)

v2_lr_logloss = log_loss(
    y_test_v2,
    v2_lr_prob,
    labels=v2_lr_classes
)


v2_lr_class_to_index = {
    class_name: index
    for index, class_name in enumerate(v2_lr_classes)
}

v2_lr_true_encoded = y_test_v2.map(
    v2_lr_class_to_index
).to_numpy()

v2_lr_brier = (
    (
        (
            v2_lr_prob -
            (
                v2_lr_true_encoded[:, None]
                == range(len(v2_lr_classes))
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()


print(
    "Accuracy:",
    round(v2_lr_accuracy, 4)
)

print(
    "Log Loss:",
    round(v2_lr_logloss, 4)
)

print(
    "Brier Score:",
    round(v2_lr_brier, 4)
)


# V2 CATBOOST
print("\n===== V2 CATBOOST =====")

catboost_model_v2 = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.03,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False
)

catboost_model_v2.fit(
    X_train_v2,
    y_train_v2
)

v2_cb_pred = catboost_model_v2.predict(
    X_test_v2
).ravel()

v2_cb_prob = catboost_model_v2.predict_proba(
    X_test_v2
)

v2_cb_classes = (
    catboost_model_v2.classes_
)


# V2 CATBOOST METRICS
v2_cb_accuracy = accuracy_score(
    y_test_v2,
    v2_cb_pred
)

v2_cb_logloss = log_loss(
    y_test_v2,
    v2_cb_prob,
    labels=v2_cb_classes
)


v2_cb_class_to_index = {
    class_name: index
    for index, class_name in enumerate(v2_cb_classes)
}

v2_cb_true_encoded = y_test_v2.map(
    v2_cb_class_to_index
).to_numpy()

v2_cb_brier = (
    (
        (
            v2_cb_prob -
            (
                v2_cb_true_encoded[:, None]
                == range(len(v2_cb_classes))
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()


print(
    "Accuracy:",
    round(v2_cb_accuracy, 4)
)

print(
    "Log Loss:",
    round(v2_cb_logloss, 4)
)

print(
    "Brier Score:",
    round(v2_cb_brier, 4)
)


# COMPARE V1.1 AND V2
print("\n===== V1.1 VS V2 COMPARISON =====")

feature_comparison = pd.DataFrame({
    "Model": [
        "V1.1 Logistic Regression",
        "V2 Logistic Regression",
        "V1.1 CatBoost",
        "V2 CatBoost"
    ],
    "Features": [
        len(model_features),
        len(model_features_v2),
        len(model_features),
        len(model_features_v2)
    ],
    "Accuracy": [
        accuracy,
        v2_lr_accuracy,
        cb_accuracy,
        v2_cb_accuracy
    ],
    "Log Loss": [
        logloss,
        v2_lr_logloss,
        cb_logloss,
        v2_cb_logloss
    ],
    "Brier Score": [
        brier_score,
        v2_lr_brier,
        cb_brier_score,
        v2_cb_brier
    ]
})

print(
    feature_comparison.round(4)
)


# V2 LOGISTIC REGRESSION CONFUSION MATRIX
print("\n===== V2 LOGISTIC REGRESSION CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test_v2,
        v2_lr_pred,
        labels=v2_lr_classes
    )
)


# V2 CATBOOST CONFUSION MATRIX
print("\n===== V2 CATBOOST CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test_v2,
        v2_cb_pred,
        labels=v2_cb_classes
    )
)





# PROBABILITY CALIBRATION
# TIME-AWARE CALIBRATION SPLITS
calibration_cv = TimeSeriesSplit(
    n_splits=5
)


# CALIBRATE LOGISTIC REGRESSION
print("\n===== CALIBRATING LOGISTIC REGRESSION =====")

calibrated_logistic = CalibratedClassifierCV(
    estimator=logistic_model,
    method="sigmoid",
    cv=calibration_cv,
    ensemble=True
)

calibrated_logistic.fit(
    X_train,
    y_train
)

calibrated_lr_pred = (
    calibrated_logistic.predict(
        X_test
    )
)

calibrated_lr_prob = (
    calibrated_logistic.predict_proba(
        X_test
    )
)

calibrated_lr_classes = (
    calibrated_logistic.classes_
)


# CALIBRATED LOGISTIC METRICS
calibrated_lr_accuracy = accuracy_score(
    y_test,
    calibrated_lr_pred
)

calibrated_lr_logloss = log_loss(
    y_test,
    calibrated_lr_prob,
    labels=calibrated_lr_classes
)

calibrated_lr_class_to_index = {
    class_name: index
    for index, class_name
    in enumerate(calibrated_lr_classes)
}

calibrated_lr_true_encoded = (
    y_test
    .map(calibrated_lr_class_to_index)
    .to_numpy()
)

calibrated_lr_brier = (
    (
        (
            calibrated_lr_prob -
            (
                calibrated_lr_true_encoded[:, None]
                == range(
                    len(calibrated_lr_classes)
                )
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()


print(
    "Accuracy:",
    round(
        calibrated_lr_accuracy,
        4
    )
)

print(
    "Log Loss:",
    round(
        calibrated_lr_logloss,
        4
    )
)

print(
    "Brier Score:",
    round(
        calibrated_lr_brier,
        4
    )
)


# CALIBRATE V2 CATBOOST
print("\n===== CALIBRATING V2 CATBOOST =====")

calibrated_catboost = CalibratedClassifierCV(
    estimator=catboost_model_v2,
    method="sigmoid",
    cv=calibration_cv,
    ensemble=True
)

calibrated_catboost.fit(
    X_train_v2,
    y_train_v2
)

calibrated_cb_pred = (
    calibrated_catboost.predict(
        X_test_v2
    )
)

calibrated_cb_prob = (
    calibrated_catboost.predict_proba(
        X_test_v2
    )
)

calibrated_cb_classes = (
    calibrated_catboost.classes_
)


# CALIBRATED CATBOOST METRICS
calibrated_cb_accuracy = accuracy_score(
    y_test_v2,
    calibrated_cb_pred
)

calibrated_cb_logloss = log_loss(
    y_test_v2,
    calibrated_cb_prob,
    labels=calibrated_cb_classes
)

calibrated_cb_class_to_index = {
    class_name: index
    for index, class_name
    in enumerate(calibrated_cb_classes)
}

calibrated_cb_true_encoded = (
    y_test_v2
    .map(calibrated_cb_class_to_index)
    .to_numpy()
)

calibrated_cb_brier = (
    (
        (
            calibrated_cb_prob -
            (
                calibrated_cb_true_encoded[:, None]
                == range(
                    len(calibrated_cb_classes)
                )
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()


print(
    "Accuracy:",
    round(
        calibrated_cb_accuracy,
        4
    )
)

print(
    "Log Loss:",
    round(
        calibrated_cb_logloss,
        4
    )
)

print(
    "Brier Score:",
    round(
        calibrated_cb_brier,
        4
    )
)


# CALIBRATION COMPARISON
print("\n===== CALIBRATION COMPARISON =====")

calibration_comparison = pd.DataFrame({
    "Model": [
        "V1.1 Logistic Regression",
        "Calibrated Logistic Regression",
        "V2 CatBoost",
        "Calibrated V2 CatBoost"
    ],
    "Accuracy": [
        accuracy,
        calibrated_lr_accuracy,
        v2_cb_accuracy,
        calibrated_cb_accuracy
    ],
    "Log Loss": [
        logloss,
        calibrated_lr_logloss,
        v2_cb_logloss,
        calibrated_cb_logloss
    ],
    "Brier Score": [
        brier_score,
        calibrated_lr_brier,
        v2_cb_brier,
        calibrated_cb_brier
    ]
})

print(
    calibration_comparison.round(4)
)





# CALIBRATED CATBOOST PROBABILITY DIAGNOSTICS
print("\n===== CALIBRATED CATBOOST DIAGNOSTICS =====")


# PREDICTED CLASS DISTRIBUTION
calibrated_cb_counts = (
    pd.Series(calibrated_cb_pred)
    .value_counts()
    .reindex(
        calibrated_cb_classes,
        fill_value=0
    )
)

print("\nPredicted class counts:")
print(calibrated_cb_counts)

print("\nPredicted class percentages:")
print(
    (
        calibrated_cb_counts /
        len(calibrated_cb_pred)
        * 100
    ).round(2)
)


# ACTUAL TEST DISTRIBUTION
actual_cb_counts = (
    y_test_v2
    .value_counts()
    .reindex(
        calibrated_cb_classes,
        fill_value=0
    )
)

print("\nActual test class counts:")
print(actual_cb_counts)

print("\nActual test class percentages:")
print(
    (
        actual_cb_counts /
        len(y_test_v2)
        * 100
    ).round(2)
)


# AVERAGE PREDICTED PROBABILITIES
calibrated_cb_average_probabilities = pd.Series(
    calibrated_cb_prob.mean(axis=0),
    index=calibrated_cb_classes
)

print("\nAverage predicted probabilities:")
print(
    calibrated_cb_average_probabilities.round(4)
)

print("\nAverage predicted probabilities (%):")
print(
    (
        calibrated_cb_average_probabilities
        * 100
    ).round(2)
)


# ACTUAL FREQUENCY VS PREDICTED PROBABILITY
calibrated_cb_probability_comparison = pd.DataFrame({
    "average_predicted_probability":
        calibrated_cb_average_probabilities,

    "actual_frequency":
        (
            y_test_v2
            .value_counts(normalize=True)
            .reindex(calibrated_cb_classes)
        )
})

print(
    "\n===== PROBABILITY VS ACTUAL FREQUENCY ====="
)

print(
    calibrated_cb_probability_comparison.round(4)
)


# DRAW PROBABILITY ANALYSIS
draw_index_cb = list(
    calibrated_cb_classes
).index("D")

calibrated_draw_probabilities = (
    calibrated_cb_prob[:, draw_index_cb]
)

print(
    "\n===== CALIBRATED DRAW PROBABILITY ====="
)

print(
    "Average draw probability:",
    round(
        calibrated_draw_probabilities.mean(),
        4
    )
)

print(
    "Maximum draw probability:",
    round(
        calibrated_draw_probabilities.max(),
        4
    )
)

print(
    "Minimum draw probability:",
    round(
        calibrated_draw_probabilities.min(),
        4
    )
)


# CONFUSION MATRIX
print(
    "\n===== CALIBRATED CATBOOST CONFUSION MATRIX ====="
)

print(
    confusion_matrix(
        y_test_v2,
        calibrated_cb_pred,
        labels=calibrated_cb_classes
    )
)


# CLASSIFICATION REPORT
print(
    "\n===== CALIBRATED CATBOOST CLASSIFICATION REPORT ====="
)

print(
    classification_report(
        y_test_v2,
        calibrated_cb_pred,
        labels=calibrated_cb_classes
    )
)

print("\n")


# FEATURE ENGINEERING V3
# DRAW AND BALANCE FEATURES
features_v3 = features_v2.copy()


# ABSOLUTE STRENGTH GAPS
features_v3["abs_win_rate_gap"] = (
    features_v3["win_rate_edge"].abs()
)

features_v3["abs_venue_win_rate_gap"] = (
    features_v3["venue_win_rate_edge"].abs()
)

features_v3["abs_recent_points_gap"] = (
    features_v3["recent_points_edge"].abs()
)

features_v3["abs_goal_scoring_gap"] = (
    features_v3["home_goal_scoring_edge"].abs()
)

features_v3["abs_defensive_gap"] = (
    features_v3["home_defensive_edge"].abs()
)


# COMBINED TEAM FORM
features_v3["combined_recent_points"] = (
    features_v3["home_recent_points_5"]
    + features_v3["away_recent_points_5"]
)

features_v3["combined_win_rate"] = (
    features_v3["home_win_rate"]
    + features_v3["away_win_rate"]
)


# COMBINED GOAL ENVIRONMENT
features_v3["combined_avg_goals_for"] = (
    features_v3["home_avg_goals_for"]
    + features_v3["away_avg_goals_for"]
)

features_v3["combined_avg_goals_against"] = (
    features_v3["home_avg_goals_against"]
    + features_v3["away_avg_goals_against"]
)


# TEAM BALANCE SCORE
# Higher value = more similar historical win rates
features_v3["win_rate_balance"] = (
    1 - features_v3["abs_win_rate_gap"]
)


# FORM BALANCE SCORE
# Higher value = more similar recent form
features_v3["recent_form_balance"] = (
    1 -
    (
        features_v3["abs_recent_points_gap"]
        / 15
    )
)


# LIST NEW FEATURES
v3_new_features = [
    "abs_win_rate_gap",
    "abs_venue_win_rate_gap",
    "abs_recent_points_gap",
    "abs_goal_scoring_gap",
    "abs_defensive_gap",
    "combined_recent_points",
    "combined_win_rate",
    "combined_avg_goals_for",
    "combined_avg_goals_against",
    "win_rate_balance",
    "recent_form_balance"
]


# DISPLAY V3 FEATURES
print("\n===== FEATURE ENGINEERING V3 =====")

print("\nNew V3 features:")

for feature in v3_new_features:
    print("-", feature)


# CHECK V3 FEATURE STATISTICS
print("\n===== V3 FEATURE STATISTICS =====")

print(
    features_v3[
        v3_new_features
    ]
    .describe()
    .round(4)
)


# CHECK MISSING VALUES
print("\n===== V3 MISSING VALUES =====")

print(
    features_v3[
        v3_new_features
    ]
    .isnull()
    .sum()
)


# CREATE V3 MODEL FEATURES
model_features_v3 = (
    model_features_v2
    + v3_new_features
)


print("\n===== V3 MODEL FEATURES =====")

print(
    "Number of features:",
    len(model_features_v3)
)


# CREATE V3 CHRONOLOGICAL DATASET
model_df_v3 = (
    features_v3
    .sort_values("date")
    .reset_index(drop=True)
)


# TIME-AWARE SPLIT
split_index_v3 = int(
    len(model_df_v3) * 0.80
)

cutoff_date_v3 = model_df_v3.loc[
    split_index_v3,
    "date"
]

train_df_v3 = model_df_v3[
    model_df_v3["date"] < cutoff_date_v3
].copy()

test_df_v3 = model_df_v3[
    model_df_v3["date"] >= cutoff_date_v3
].copy()


# CREATE TRAINING / TEST FEATURES
X_train_v3 = train_df_v3[
    model_features_v3
]

y_train_v3 = train_df_v3[
    "result"
]

X_test_v3 = test_df_v3[
    model_features_v3
]

y_test_v3 = test_df_v3[
    "result"
]


# V3 SPLIT VALIDATION
print("\n===== V3 TRAIN / TEST SPLIT =====")

print(
    "Training rows:",
    len(train_df_v3)
)

print(
    "Testing rows:",
    len(test_df_v3)
)

print(
    "Training dates:",
    train_df_v3["date"].min(),
    "to",
    train_df_v3["date"].max()
)

print(
    "Testing dates:",
    test_df_v3["date"].min(),
    "to",
    test_df_v3["date"].max()
)


# CHECK DATE OVERLAP
v3_overlapping_dates = set(
    train_df_v3["date"]
).intersection(
    set(test_df_v3["date"])
)

print(
    "\nOverlapping dates:",
    len(v3_overlapping_dates)
)

if (
    train_df_v3["date"].max()
    < test_df_v3["date"].min()
    and
    len(v3_overlapping_dates) == 0
):
    print(
        "V3 temporal split verified: PASS"
    )
else:
    print(
        "V3 temporal split verified: FAIL"
    )


# SAVE V3
features_v3_output = (
    PROCESSED_DIR /
    "champions_league_features_v3.csv"
)

features_v3.to_csv(
    features_v3_output,
    index=False
)

print(
    "\nFeature dataset V3 saved successfully."
)

print(
    "Saved to:",
    features_v3_output
)




# V3 MODEL TEST
# CATBOOST + LOGISTIC REGRESSION
print("\n===== V3 LOGISTIC REGRESSION =====")

v3_logistic_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)

v3_logistic_model.fit(
    X_train_v3,
    y_train_v3
)

v3_lr_pred = v3_logistic_model.predict(
    X_test_v3
)

v3_lr_prob = v3_logistic_model.predict_proba(
    X_test_v3
)

v3_lr_classes = (
    v3_logistic_model
    .named_steps["model"]
    .classes_
)

v3_lr_accuracy = accuracy_score(
    y_test_v3,
    v3_lr_pred
)

v3_lr_logloss = log_loss(
    y_test_v3,
    v3_lr_prob,
    labels=v3_lr_classes
)

v3_lr_class_to_index = {
    class_name: index
    for index, class_name in enumerate(
        v3_lr_classes
    )
}

v3_lr_true_encoded = (
    y_test_v3
    .map(v3_lr_class_to_index)
    .to_numpy()
)

v3_lr_brier = (
    (
        (
            v3_lr_prob -
            (
                v3_lr_true_encoded[:, None]
                == range(
                    len(v3_lr_classes)
                )
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()

print(
    "Accuracy:",
    round(v3_lr_accuracy, 4)
)

print(
    "Log Loss:",
    round(v3_lr_logloss, 4)
)

print(
    "Brier Score:",
    round(v3_lr_brier, 4)
)


# V3 CATBOOST
print("\n===== V3 CATBOOST =====")

v3_catboost_model = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.03,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False
)

v3_catboost_model.fit(
    X_train_v3,
    y_train_v3
)

v3_cb_pred = (
    v3_catboost_model
    .predict(X_test_v3)
    .ravel()
)

v3_cb_prob = (
    v3_catboost_model
    .predict_proba(X_test_v3)
)

v3_cb_classes = (
    v3_catboost_model.classes_
)

v3_cb_accuracy = accuracy_score(
    y_test_v3,
    v3_cb_pred
)

v3_cb_logloss = log_loss(
    y_test_v3,
    v3_cb_prob,
    labels=v3_cb_classes
)

v3_cb_class_to_index = {
    class_name: index
    for index, class_name in enumerate(
        v3_cb_classes
    )
}

v3_cb_true_encoded = (
    y_test_v3
    .map(v3_cb_class_to_index)
    .to_numpy()
)

v3_cb_brier = (
    (
        (
            v3_cb_prob -
            (
                v3_cb_true_encoded[:, None]
                == range(
                    len(v3_cb_classes)
                )
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()

print(
    "Accuracy:",
    round(v3_cb_accuracy, 4)
)

print(
    "Log Loss:",
    round(v3_cb_logloss, 4)
)

print(
    "Brier Score:",
    round(v3_cb_brier, 4)
)


# V3 MODEL COMPARISON
print("\n===== V3 MODEL COMPARISON =====")

v3_comparison = pd.DataFrame({
    "Model": [
        "V3 Logistic Regression",
        "V3 CatBoost",
        "Calibrated V2 CatBoost"
    ],
    "Accuracy": [
        v3_lr_accuracy,
        v3_cb_accuracy,
        calibrated_cb_accuracy
    ],
    "Log Loss": [
        v3_lr_logloss,
        v3_cb_logloss,
        calibrated_cb_logloss
    ],
    "Brier Score": [
        v3_lr_brier,
        v3_cb_brier,
        calibrated_cb_brier
    ]
})

print(
    v3_comparison.round(4)
)


# CALIBRATE V3 CATBOOST
print("\n===== CALIBRATING V3 CATBOOST =====")

v3_calibration_cv = TimeSeriesSplit(
    n_splits=5
)

calibrated_v3_catboost = (
    CalibratedClassifierCV(
        estimator=v3_catboost_model,
        method="sigmoid",
        cv=v3_calibration_cv,
        ensemble=True
    )
)

calibrated_v3_catboost.fit(
    X_train_v3,
    y_train_v3
)

v3_calibrated_cb_pred = (
    calibrated_v3_catboost
    .predict(X_test_v3)
)

v3_calibrated_cb_prob = (
    calibrated_v3_catboost
    .predict_proba(X_test_v3)
)

v3_calibrated_cb_classes = (
    calibrated_v3_catboost.classes_
)


# V3 CALIBRATED METRICS
v3_calibrated_accuracy = accuracy_score(
    y_test_v3,
    v3_calibrated_cb_pred
)

v3_calibrated_logloss = log_loss(
    y_test_v3,
    v3_calibrated_cb_prob,
    labels=v3_calibrated_cb_classes
)

v3_calibrated_class_to_index = {
    class_name: index
    for index, class_name in enumerate(
        v3_calibrated_cb_classes
    )
}

v3_calibrated_true_encoded = (
    y_test_v3
    .map(v3_calibrated_class_to_index)
    .to_numpy()
)

v3_calibrated_brier = (
    (
        (
            v3_calibrated_cb_prob -
            (
                v3_calibrated_true_encoded[:, None]
                == range(
                    len(v3_calibrated_cb_classes)
                )
            ).astype(float)
        ) ** 2
    ).sum(axis=1)
).mean()

print(
    "Accuracy:",
    round(v3_calibrated_accuracy, 4)
)

print(
    "Log Loss:",
    round(v3_calibrated_logloss, 4)
)

print(
    "Brier Score:",
    round(v3_calibrated_brier, 4)
)


# FINAL V2 VS V3 CALIBRATION COMPARISON
print("\n===== V2 VS V3 CALIBRATED CATBOOST =====")

final_feature_comparison = pd.DataFrame({
    "Model": [
        "Calibrated V2 CatBoost",
        "Calibrated V3 CatBoost"
    ],
    "Features": [
        len(model_features_v2),
        len(model_features_v3)
    ],
    "Accuracy": [
        calibrated_cb_accuracy,
        v3_calibrated_accuracy
    ],
    "Log Loss": [
        calibrated_cb_logloss,
        v3_calibrated_logloss
    ],
    "Brier Score": [
        calibrated_cb_brier,
        v3_calibrated_brier
    ]
})

print(
    final_feature_comparison.round(4)
)





# SEASON-BY-SEASON ROBUSTNESS TEST
# CALIBRATED V2 CATBOOST
print("\n===== SEASON-BY-SEASON ROBUSTNESS =====")


# CREATE TEST RESULTS DATAFRAME
season_results = test_df_v2[
    [
        "season",
        "date",
        "home_team",
        "away_team",
        "result"
    ]
].copy()

season_results["predicted_result"] = (
    calibrated_cb_pred
)

season_results["away_probability"] = (
    calibrated_cb_prob[
        :,
        list(calibrated_cb_classes).index("A")
    ]
)

season_results["draw_probability"] = (
    calibrated_cb_prob[
        :,
        list(calibrated_cb_classes).index("D")
    ]
)

season_results["home_probability"] = (
    calibrated_cb_prob[
        :,
        list(calibrated_cb_classes).index("H")
    ]
)


# EVALUATE EACH SEASON
season_metrics = []

for season in sorted(
    season_results["season"].unique()
):

    season_data = season_results[
        season_results["season"] == season
    ]

    season_accuracy = accuracy_score(
        season_data["result"],
        season_data["predicted_result"]
    )

    season_logloss = log_loss(
        season_data["result"],
        season_data[
            [
                "away_probability",
                "draw_probability",
                "home_probability"
            ]
        ],
        labels=["A", "D", "H"]
    )

    season_true_encoded = (
        season_data["result"]
        .map({
            "A": 0,
            "D": 1,
            "H": 2
        })
        .to_numpy()
    )

    season_probabilities = (
        season_data[
            [
                "away_probability",
                "draw_probability",
                "home_probability"
            ]
        ]
        .to_numpy()
    )

    season_brier = (
        (
            (
                season_probabilities -
                (
                    season_true_encoded[:, None]
                    == range(3)
                ).astype(float)
            ) ** 2
        )
        .sum(axis=1)
        .mean()
    )

    season_metrics.append({
        "season": season,
        "matches": len(season_data),
        "accuracy": season_accuracy,
        "log_loss": season_logloss,
        "brier_score": season_brier
    })


# DISPLAY SEASON METRICS
season_metrics_df = pd.DataFrame(
    season_metrics
)

print(
    season_metrics_df.round(4)
)


# TEST SET OVERALL RESULT
print("\n===== OVERALL TEST SET =====")

print(
    "Accuracy:",
    round(calibrated_cb_accuracy, 4)
)

print(
    "Log Loss:",
    round(calibrated_cb_logloss, 4)
)

print(
    "Brier Score:",
    round(calibrated_cb_brier, 4)
)




# SAVE FINAL MODEL ARTIFACTS
# CREATE MODEL DIRECTORY
MODEL_DIR = Path(
    "data/models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# SAVE THE EVALUATION MODEL
# This is the exact model that produced our benchmark:
#
# Accuracy:   58.70%
# Log Loss:    0.9388
# Brier:       0.5546

evaluation_model_path = (
    MODEL_DIR /
    "calibrated_v2_catboost_evaluation.joblib"
)

joblib.dump(
    calibrated_catboost,
    evaluation_model_path
)

print(
    "\nEvaluation model saved to:"
)

print(
    evaluation_model_path
)


# RETRAIN SELECTED MODEL ON FULL V2 DATA
print(
    "\n===== TRAINING PRODUCTION MODEL ====="
)

X_full_v2 = features_v2[
    model_features_v2
]

y_full_v2 = features_v2[
    "result"
]


# CREATE TIME-AWARE CALIBRATION
production_calibration_cv = TimeSeriesSplit(
    n_splits=5
)


# CREATE BASE CATBOOST MODEL
production_catboost = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.03,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False
)


# CREATE CALIBRATED PRODUCTION MODEL
production_model = CalibratedClassifierCV(
    estimator=production_catboost,
    method="sigmoid",
    cv=production_calibration_cv,
    ensemble=True
)


# TRAIN PRODUCTION MODEL
production_model.fit(
    X_full_v2,
    y_full_v2
)

print(
    "Production model training completed successfully."
)


# CHECK MODEL CLASSES
production_classes = (
    production_model.classes_
)

print(
    "\nProduction model classes:"
)

print(
    production_classes
)


# SAVE PRODUCTION MODEL
production_model_path = (
    MODEL_DIR /
    "champions_league_calibrated_v2_catboost.joblib"
)

joblib.dump(
    production_model,
    production_model_path
)

print(
    "\nProduction model saved to:"
)

print(
    production_model_path
)


# SAVE FEATURE CONFIGURATION
feature_config = {
    "feature_version": "V2",
    "number_of_features": len(
        model_features_v2
    ),
    "features": model_features_v2
}

feature_config_path = (
    MODEL_DIR /
    "champions_league_v2_feature_config.json"
)

with open(
    feature_config_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        feature_config,
        file,
        indent=4
    )

print(
    "\nFeature configuration saved to:"
)

print(
    feature_config_path
)


# SAVE MODEL METADATA
model_metadata = {
    "model_name": (
        "Calibrated CatBoost"
    ),

    "feature_version": "V2",

    "training_matches": len(
        features_v2
    ),

    "historical_period": {
        "start": str(
            features_v2["date"].min().date()
        ),
        "end": str(
            features_v2["date"].max().date()
        )
    },

    "classes": [
        str(class_name)
        for class_name in production_classes
    ],

    "catboost_parameters": {
        "iterations": 500,
        "depth": 5,
        "learning_rate": 0.03,
        "loss_function": "MultiClass",
        "random_seed": 42
    },

    "calibration": {
        "method": "sigmoid",
        "cv": "TimeSeriesSplit",
        "n_splits": 5,
        "ensemble": True
    },

    "evaluation_results": {
        "test_period": (
            "2023-11-29 to 2026-05-30"
        ),
        "accuracy": 0.5870,
        "log_loss": 0.9388,
        "brier_score": 0.5546
    }
}

metadata_path = (
    MODEL_DIR /
    "champions_league_model_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_metadata,
        file,
        indent=4
    )

print(
    "\nModel metadata saved to:"
)

print(
    metadata_path
)


# RELOAD MODEL TO VERIFY SAVING
print(
    "\n===== MODEL SAVE VERIFICATION ====="
)

loaded_model = joblib.load(
    production_model_path
)

print(
    "Model successfully reloaded."
)


# TEST LOADED MODEL
sample_probabilities = (
    loaded_model.predict_proba(
        X_full_v2.head(1)
    )
)

print(
    "\nSample probability output:"
)

print(
    sample_probabilities
)

print(
    "\nModel artifact verification: PASS"
)



print("\n PRODUCTION PREDICTION ENGINE")


# PRODUCTION PREDICTION ENGINE
# LOAD PRODUCTION MODEL
production_model = joblib.load(
    "data/models/champions_league_calibrated_v2_catboost.joblib"
)


# LOAD FEATURE CONFIGURATION
with open(
    "data/models/champions_league_v2_feature_config.json",
    "r",
    encoding="utf-8"
) as file:

    feature_config = json.load(file)


model_features_v2 = feature_config["features"]


# LOAD HISTORICAL MATCH DATA
historical_matches = pd.read_csv(
    "data/processed/champions_league_all_seasons_clean.csv"
)

historical_matches["date"] = pd.to_datetime(
    historical_matches["date"]
)

historical_matches = historical_matches.sort_values(
    "date"
).reset_index(drop=True)


print(
    "\n===== PRODUCTION PREDICTION ENGINE ====="
)

print(
    "Production model loaded successfully."
)

print(
    "Historical matches loaded:",
    len(historical_matches)
)

print(
    "Number of model features:",
    len(model_features_v2)
)


# CREATE EMPTY TEAM STATISTICS
def create_prediction_team_stats():

    return {
        "matches": 0,

        "goals_for": 0,
        "goals_against": 0,

        "wins": 0,
        "draws": 0,
        "losses": 0,

        "home_matches": 0,
        "home_wins": 0,

        "away_matches": 0,
        "away_wins": 0,

        "recent_points": deque(
            maxlen=5
        ),

        "last_match_date": None
    }


# BUILD PRE-MATCH TEAM STATISTICS
def build_team_stats_before_date(
    match_date
):

    match_date = pd.to_datetime(
        match_date
    )

    # USE ONLY MATCHES BEFORE THE PREDICTION DATE
    previous_matches = historical_matches[
        historical_matches["date"] < match_date
    ].copy()

    team_stats = {}


    # PROCESS HISTORICAL MATCHES
    for _, row in previous_matches.iterrows():

        home_team = row["home_team"]
        away_team = row["away_team"]

        home_goals = int(
            row["home_goals"]
        )

        away_goals = int(
            row["away_goals"]
        )

        result = row["result"]


        # CREATE TEAM RECORDS IF NECESSARY
        if home_team not in team_stats:

            team_stats[home_team] = (
                create_prediction_team_stats()
            )

        if away_team not in team_stats:

            team_stats[away_team] = (
                create_prediction_team_stats()
            )


        home_stats = team_stats[
            home_team
        ]

        away_stats = team_stats[
            away_team
        ]


        # GENERAL MATCH STATISTICS
        home_stats["matches"] += 1
        away_stats["matches"] += 1


        home_stats["goals_for"] += (
            home_goals
        )

        home_stats["goals_against"] += (
            away_goals
        )


        away_stats["goals_for"] += (
            away_goals
        )

        away_stats["goals_against"] += (
            home_goals
        )


        # VENUE STATISTICS
        home_stats["home_matches"] += 1

        away_stats["away_matches"] += 1


        # RESULT STATISTICS
        if result == "H":

            home_stats["wins"] += 1
            away_stats["losses"] += 1

            home_stats["home_wins"] += 1

            home_stats[
                "recent_points"
            ].append(3)

            away_stats[
                "recent_points"
            ].append(0)


        elif result == "A":

            home_stats["losses"] += 1
            away_stats["wins"] += 1

            away_stats["away_wins"] += 1

            home_stats[
                "recent_points"
            ].append(0)

            away_stats[
                "recent_points"
            ].append(3)


        else:

            home_stats["draws"] += 1
            away_stats["draws"] += 1

            home_stats[
                "recent_points"
            ].append(1)

            away_stats[
                "recent_points"
            ].append(1)


        # LAST MATCH DATE
        home_stats[
            "last_match_date"
        ] = row["date"]

        away_stats[
            "last_match_date"
        ] = row["date"]


    return team_stats


# PREDICT A NEW MATCH
def predict_match(
    home_team,
    away_team,
    match_date
):

    # DEBUG: CONFIRM FUNCTION STARTED
    print(
        "\n===== PREDICT_MATCH FUNCTION STARTED ====="
    )

    print(
        "Home team:",
        home_team
    )

    print(
        "Away team:",
        away_team
    )

    print(
        "Match date:",
        match_date
    )


    # CONVERT DATE
    match_date = pd.to_datetime(
        match_date
    )


    # BUILD PRE-MATCH TEAM STATISTICS
    team_stats = build_team_stats_before_date(
        match_date
    )

    print(
        "Historical team statistics built successfully."
    )

    print(
        "Teams available:",
        len(team_stats)
    )


    # CREATE EMPTY RECORD FOR UNKNOWN HOME TEAM
    if home_team not in team_stats:

        team_stats[home_team] = (
            create_prediction_team_stats()
        )

        print(
            f"Warning: {home_team} has no "
            "previous Champions League history "
            "in the dataset."
        )


    # CREATE EMPTY RECORD FOR UNKNOWN AWAY TEAM
    if away_team not in team_stats:

        team_stats[away_team] = (
            create_prediction_team_stats()
        )

        print(
            f"Warning: {away_team} has no "
            "previous Champions League history "
            "in the dataset."
        )


    # GET TEAM STATISTICS
    home_stats = team_stats[
        home_team
    ]

    away_stats = team_stats[
        away_team
    ]


    # BASIC PRE-MATCH FEATURES
    home_matches = home_stats[
        "matches"
    ]

    away_matches = away_stats[
        "matches"
    ]


    # AVERAGE GOALS FOR
    home_avg_goals_for = (
        home_stats["goals_for"]
        / home_matches
        if home_matches > 0
        else 0
    )

    away_avg_goals_for = (
        away_stats["goals_for"]
        / away_matches
        if away_matches > 0
        else 0
    )


    # AVERAGE GOALS AGAINST
    home_avg_goals_against = (
        home_stats["goals_against"]
        / home_matches
        if home_matches > 0
        else 0
    )

    away_avg_goals_against = (
        away_stats["goals_against"]
        / away_matches
        if away_matches > 0
        else 0
    )


    # WIN RATE
    home_win_rate = (
        home_stats["wins"]
        / home_matches
        if home_matches > 0
        else 0
    )

    away_win_rate = (
        away_stats["wins"]
        / away_matches
        if away_matches > 0
        else 0
    )


    # VENUE WIN RATE
    home_home_win_rate = (
        home_stats["home_wins"]
        / home_stats["home_matches"]
        if home_stats["home_matches"] > 0
        else 0
    )

    away_away_win_rate = (
        away_stats["away_wins"]
        / away_stats["away_matches"]
        if away_stats["away_matches"] > 0
        else 0
    )


    # RECENT FORM
    home_recent_points_5 = sum(
        home_stats["recent_points"]
    )

    away_recent_points_5 = sum(
        away_stats["recent_points"]
    )


    # DAYS SINCE LAST MATCH
    if (
        home_stats["last_match_date"]
        is None
    ):

        home_days_since_match = -1

    else:

        home_days_since_match = (
            match_date
            - home_stats["last_match_date"]
        ).days


    if (
        away_stats["last_match_date"]
        is None
    ):

        away_days_since_match = -1

    else:

        away_days_since_match = (
            match_date
            - away_stats["last_match_date"]
        ).days


    # CAP DAYS SINCE LAST MATCH
    home_days_since_match_capped = max(
        0,
        min(
            home_days_since_match,
            365
        )
    )

    away_days_since_match_capped = max(
        0,
        min(
            away_days_since_match,
            365
        )
    )


    # PREVIOUS MATCH FLAGS
    home_has_previous_match = int(
        home_days_since_match >= 0
    )

    away_has_previous_match = int(
        away_days_since_match >= 0
    )


    # V2 RELATIVE FEATURES
    home_goal_scoring_edge = (
        home_avg_goals_for
        - away_avg_goals_for
    )

    home_defensive_edge = (
        away_avg_goals_against
        - home_avg_goals_against
    )

    home_attack_vs_away_defense = (
        home_avg_goals_for
        - away_avg_goals_against
    )

    away_attack_vs_home_defense = (
        away_avg_goals_for
        - home_avg_goals_against
    )

    win_rate_edge = (
        home_win_rate
        - away_win_rate
    )

    venue_win_rate_edge = (
        home_home_win_rate
        - away_away_win_rate
    )

    recent_points_edge = (
        home_recent_points_5
        - away_recent_points_5
    )

    experience_edge = (
        home_matches
        - away_matches
    )

    days_since_match_edge = (
        home_days_since_match_capped
        - away_days_since_match_capped
    )


    # CREATE PREDICTION DATAFRAME
    prediction_row = pd.DataFrame([
        {

            "home_matches_before":
                home_matches,

            "away_matches_before":
                away_matches,


            "home_avg_goals_for":
                home_avg_goals_for,

            "home_avg_goals_against":
                home_avg_goals_against,


            "away_avg_goals_for":
                away_avg_goals_for,

            "away_avg_goals_against":
                away_avg_goals_against,


            "home_win_rate":
                home_win_rate,

            "away_win_rate":
                away_win_rate,


            "home_home_win_rate":
                home_home_win_rate,

            "away_away_win_rate":
                away_away_win_rate,


            "home_recent_points_5":
                home_recent_points_5,

            "away_recent_points_5":
                away_recent_points_5,


            "home_days_since_match_capped":
                home_days_since_match_capped,

            "away_days_since_match_capped":
                away_days_since_match_capped,


            "home_has_previous_match":
                home_has_previous_match,

            "away_has_previous_match":
                away_has_previous_match,


            "home_goal_scoring_edge":
                home_goal_scoring_edge,

            "home_defensive_edge":
                home_defensive_edge,


            "home_attack_vs_away_defense":
                home_attack_vs_away_defense,

            "away_attack_vs_home_defense":
                away_attack_vs_home_defense,


            "win_rate_edge":
                win_rate_edge,

            "venue_win_rate_edge":
                venue_win_rate_edge,


            "recent_points_edge":
                recent_points_edge,

            "experience_edge":
                experience_edge,

            "days_since_match_edge":
                days_since_match_edge

        }
    ])


    # FORCE EXACT MODEL FEATURE ORDER
    prediction_row = prediction_row[
        model_features_v2
    ]


    # DEBUG: VERIFY PREDICTION DATA
    print(
        "\nPrediction feature row created successfully."
    )

    print(
        "Prediction feature shape:",
        prediction_row.shape
    )

    print(
        "Expected feature count:",
        len(model_features_v2)
    )


    # GENERATE PROBABILITIES
    probabilities = (
        production_model
        .predict_proba(
            prediction_row
        )[0]
    )

    classes = (
        production_model.classes_
    )


    # MAP CLASSES TO PROBABILITIES
    probability_map = dict(
        zip(
            classes,
            probabilities
        )
    )


    # EXTRACT H / D / A PROBABILITIES
    home_probability = (
        probability_map["H"] * 100
    )

    draw_probability = (
        probability_map["D"] * 100
    )

    away_probability = (
        probability_map["A"] * 100
    )


    # DETERMINE HIGHEST-PROBABILITY RESULT
    predicted_result = max(
        probability_map,
        key=probability_map.get
    )


    result_names = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win"
    }


    # DISPLAY PREDICTION
    print(
        "\n========================================"
    )

    print(
        "CHAMPIONS LEAGUE MATCH PREDICTION"
    )

    print(
        "========================================"
    )


    print(
        f"Home: {home_team}"
    )

    print(
        f"Away: {away_team}"
    )

    print(
        f"Date: {match_date.date()}"
    )


    print(
        "\nProbabilities:"
    )


    print(
        f"Home Win: {home_probability:.2f}%"
    )

    print(
        f"Draw:     {draw_probability:.2f}%"
    )

    print(
        f"Away Win: {away_probability:.2f}%"
    )


    print(
        "\nHighest probability:"
    )

    print(
        f"{result_names[predicted_result]}"
    )


    # RETURN PREDICTION
    return {
    "home_team": home_team,
    "away_team": away_team,
    "date": str(match_date.date()),
    "home_win_probability": float(home_probability),
    "draw_probability": float(draw_probability),
    "away_win_probability": float(away_probability),
    "predicted_result": predicted_result,
    "predicted_result_name": result_names[predicted_result]
}


# TEST PRODUCTION PREDICTION ENGINE
#print(
#    "\n===== STARTING PREDICTION TEST ====="
#)


#prediction = predict_match(
#    home_team="Real Madrid",
#    away_team="Manchester City",
#    match_date="2026-09-20"
#)


#print(
#    "\n===== PREDICTION TEST COMPLETED ====="
#)


#print(
#    "\nReturned prediction dictionary:"
#)

#print(
#    prediction
#)


print("\n PREDICTION TRACKING SYSTEM")

# PREDICTION TRACKING SYSTEM
# CREATE PREDICTION DIRECTORY
PREDICTION_DIR = Path(
    "data/predictions"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# PREDICTION LOG FILE
PREDICTION_LOG = (
    PREDICTION_DIR /
    "champions_league_predictions.csv"
)


# CREATE PREDICTION RECORD
# SAVE PREDICTION
# PREVENT DUPLICATE FIXTURES
def save_prediction(
    prediction
):

    # CREATE PREDICTION ID
    prediction_record = {
        "prediction_id": str(
            uuid.uuid4()
        ),

        "date":
            prediction["date"],

        "home_team":
            prediction["home_team"],

        "away_team":
            prediction["away_team"],

        "home_win_probability":
            float(
                prediction[
                    "home_win_probability"
                ]
            ),

        "draw_probability":
            float(
                prediction[
                    "draw_probability"
                ]
            ),

        "away_win_probability":
            float(
                prediction[
                    "away_win_probability"
                ]
            ),

        "predicted_result":
            prediction["predicted_result"],

        "predicted_result_name":
            prediction[
                "predicted_result_name"
            ],

        "actual_result":
            "",

        "status":
            "Pending"
    }


    # LOAD EXISTING PREDICTIONS
    if PREDICTION_LOG.exists():

        predictions_df = pd.read_csv(
            PREDICTION_LOG
        )

        # Make sure text columns remain text
        predictions_df[
            "actual_result"
        ] = predictions_df[
            "actual_result"
        ].astype("object")

        predictions_df[
            "status"
        ] = predictions_df[
            "status"
        ].astype("object")


        # CHECK FOR EXISTING FIXTURE
        duplicate_mask = (
            (predictions_df["date"].astype(str)
             == str(prediction["date"]))
            &
            (predictions_df["home_team"]
             == prediction["home_team"])
            &
            (predictions_df["away_team"]
             == prediction["away_team"])
        )


        if duplicate_mask.any():

            existing_record = (
                predictions_df[
                    duplicate_mask
                ].iloc[0]
            )

            print(
                "\nPrediction already exists for this fixture."
            )

            print(
                "Prediction ID:",
                existing_record[
                    "prediction_id"
                ]
            )

            print(
                "Status:",
                existing_record["status"]
            )

            return existing_record.to_dict()


    # SAVE NEW PREDICTION
    prediction_df = pd.DataFrame([
        prediction_record
    ])


    if not PREDICTION_LOG.exists():

        prediction_df.to_csv(
            PREDICTION_LOG,
            index=False
        )

    else:

        prediction_df.to_csv(
            PREDICTION_LOG,
            mode="a",
            header=False,
            index=False
        )


    print(
        "\nPrediction saved successfully."
    )

    print(
        "Prediction ID:",
        prediction_record[
            "prediction_id"
        ]
    )

    print(
        "Saved to:",
        PREDICTION_LOG
    )


    return prediction_record



# CLEAN DUPLICATE TEST PREDICTIONS
#if PREDICTION_LOG.exists():

#    predictions_df = pd.read_csv(
        #PREDICTION_LOG
    #)

#    predictions_df = predictions_df.drop_duplicates(
#        subset=[
#            "date",
#            "home_team",
#            "away_team"
#        ],
#        keep="first"
#    )

#    predictions_df.to_csv(
#        PREDICTION_LOG,
#        index=False
#    )

#    print(
#        "\nDuplicate prediction fixtures removed."
#    )


# TEST PREDICTION TRACKING
#saved_prediction = save_prediction(
    #prediction
#)

#print(
    #"\n===== SAVED PREDICTION ====="
#)

#print(
#    saved_prediction
#)


# UPDATE PREDICTION WITH ACTUAL RESULT
def update_prediction_result(
    prediction_id,
    actual_result
):

    # VALIDATE ACTUAL RESULT
    valid_results = [
        "H",
        "D",
        "A"
    ]

    actual_result = (
        str(actual_result)
        .strip()
        .upper()
    )

    if actual_result not in valid_results:

        raise ValueError(
            "Actual result must be H, D, or A."
        )


    # CHECK PREDICTION LOG EXISTS
    if not PREDICTION_LOG.exists():

        raise FileNotFoundError(
            "Prediction log does not exist."
        )


    # LOAD PREDICTION LOG
    predictions_df = pd.read_csv(
        PREDICTION_LOG
    )


    # FORCE TEXT COLUMNS
    predictions_df["actual_result"] = (
        predictions_df["actual_result"]
        .astype("object")
    )

    predictions_df["status"] = (
        predictions_df["status"]
        .astype("object")
    )


    # FIND PREDICTION
    match_mask = (
        predictions_df["prediction_id"]
        == prediction_id
    )

    if not match_mask.any():

        raise ValueError(
            "Prediction ID not found."
        )


    # CHECK IF ALREADY COMPLETED
    existing_status = (
        predictions_df.loc[
            match_mask,
            "status"
        ].iloc[0]
    )

    if existing_status == "Completed":

        print(
            "\nWarning: This prediction has "
            "already been completed."
        )


    # UPDATE ACTUAL RESULT
    predictions_df.loc[
        match_mask,
        "actual_result"
    ] = actual_result


    # UPDATE STATUS
    predictions_df.loc[
        match_mask,
        "status"
    ] = "Completed"


    # SAVE UPDATED LOG
    predictions_df.to_csv(
        PREDICTION_LOG,
        index=False
    )


    # GET UPDATED RECORD
    updated_record = (
        predictions_df.loc[
            match_mask
        ].iloc[0]
    )


    # DISPLAY RESULT
    print(
        "\nPrediction updated successfully."
    )

    print(
        "Prediction ID:",
        prediction_id
    )

    print(
        "Actual result:",
        actual_result
    )

    print(
        "Status: Completed"
    )


    return updated_record


# TEST ACTUAL RESULT UPDATE
#print(
#    "\n===== TEST ACTUAL RESULT UPDATE ====="
#)

# Use the existing completed test prediction
#updated_prediction = update_prediction_result(
#    prediction_id="61e1b5e5-19ff-488c-b055-8c78766d7080",
#    actual_result="H"
#)

#print("\n===== UPDATED PREDICTION =====")

#print(updated_prediction)



# PREDICTION EVALUATION ENGINE
# EVALUATE COMPLETED PREDICTIONS
def evaluate_predictions():

    print("\n===== PREDICTION EVALUATION ENGINE =====")


    # CHECK LOG EXISTS
    if not PREDICTION_LOG.exists():

        print(
            "Prediction log does not exist yet."
        )

        return None


    # LOAD PREDICTIONS
    predictions_df = pd.read_csv(
        PREDICTION_LOG
    )


    # COUNT PENDING / COMPLETED
    completed_df = predictions_df[
        predictions_df["status"] == "Completed"
    ].copy()

    pending_df = predictions_df[
        predictions_df["status"] == "Pending"
    ].copy()


    print(
        "\nTotal predictions:",
        len(predictions_df)
    )

    print(
        "Completed predictions:",
        len(completed_df)
    )

    print(
        "Pending predictions:",
        len(pending_df)
    )


    # STOP IF THERE ARE NOT ENOUGH COMPLETED PREDICTIONS
    if completed_df.empty:

        print(
            "\nNo completed predictions available "
            "for evaluation."
        )

        return None


    # GET ACTUAL RESULTS
    y_actual = (
        completed_df[
            "actual_result"
        ]
        .astype(str)
        .str.upper()
    )


    # GET HARD PREDICTIONS
    y_predicted = (
        completed_df[
            "predicted_result"
        ]
        .astype(str)
        .str.upper()
    )


    # GET PROBABILITIES
    probability_columns = [
        "away_win_probability",
        "draw_probability",
        "home_win_probability"
    ]

    probability_values = (
        completed_df[
            probability_columns
        ]
        .astype(float)
        .to_numpy()
    )

    y_probabilities = (
        probability_values / 100
    )


    # ACCURACY
    evaluation_accuracy = accuracy_score(
        y_actual,
        y_predicted
    )


    # LOG LOSS
    evaluation_log_loss = log_loss(
        y_actual,
        y_probabilities,
        labels=["A", "D", "H"]
    )


    # BRIER SCORE
    result_to_index = {
        "A": 0,
        "D": 1,
        "H": 2
    }

    y_actual_encoded = (
        y_actual
        .map(result_to_index)
        .to_numpy()
    )


    evaluation_brier = (
        (
            (
                y_probabilities -
                (
                    y_actual_encoded[:, None]
                    == range(3)
                ).astype(float)
            ) ** 2
        )
        .sum(axis=1)
        .mean()
    )


    # NORMALIZED BRIER SCORE
    normalized_brier = (
        evaluation_brier / 3
    )


    # CONFUSION MATRIX
    evaluation_cm = confusion_matrix(
        y_actual,
        y_predicted,
        labels=["A", "D", "H"]
    )


    # DISPLAY RESULTS
    print(
        "\n===== COMPLETED PREDICTION PERFORMANCE ====="
    )

    print(
        "Accuracy:",
        round(
            evaluation_accuracy,
            4
        )
    )

    print(
        "Log Loss:",
        round(
            evaluation_log_loss,
            4
        )
    )

    print(
        "Brier Score:",
        round(
            evaluation_brier,
            4
        )
    )

    print(
        "Normalized Brier Score:",
        round(
            normalized_brier,
            4
        )
    )


    # CONFUSION MATRIX DISPLAY
    print("\n===== CONFUSION MATRIX =====")

    print( "Classes: ['A', 'D', 'H']")

    print(evaluation_cm)


    # RESULT DISTRIBUTION
    print("\n===== ACTUAL RESULT DISTRIBUTION =====")

    print(y_actual.value_counts())


    # RETURN METRICS
    return {
        "completed_predictions":
            len(completed_df),

        "pending_predictions":
            len(pending_df),

        "accuracy":
            float(evaluation_accuracy),

        "log_loss":
            float(evaluation_log_loss),

        "brier_score":
            float(evaluation_brier),

        "normalized_brier_score":
            float(normalized_brier)
    }


# TEST EVALUATION ENGINE
#evaluation_results = evaluate_predictions()


#print(
#    "\n===== EVALUATION RESULTS ====="
#)

#print(
#    evaluation_results
#)


print("\n HISTORICAL OUT-OF-SAMPLE BACKTEST")


# HISTORICAL OUT-OF-SAMPLE BACKTEST
# Evaluate the saved evaluation model on the held-out test period
# without adding historical matches to the live prediction log.


# CREATE BACKTEST DIRECTORY
BACKTEST_DIR = Path(
    "data/predictions"
)

BACKTEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# LOAD THE SAVED EVALUATION MODEL
evaluation_model_path = (
    "data/models/calibrated_v2_catboost_evaluation.joblib"
)

backtest_model = joblib.load(
    evaluation_model_path
)

print(
    "\n===== HISTORICAL OUT-OF-SAMPLE BACKTEST ====="
)

print(
    "Evaluation model loaded successfully."
)


# PREPARE HELD-OUT TEST DATA
backtest_df = (
    test_df_v2
    .sort_values("date")
    .reset_index(drop=True)
    .copy()
)


# CREATE TEST FEATURES
X_backtest = backtest_df[
    model_features_v2
]

y_backtest = backtest_df[
    "result"
]


print(
    "\nBacktest matches:",
    len(backtest_df)
)

print(
    "Backtest date range:",
    backtest_df["date"].min(),
    "to",
    backtest_df["date"].max()
)


# GENERATE OUT-OF-SAMPLE PREDICTIONS
backtest_probabilities = (
    backtest_model.predict_proba(
        X_backtest
    )
)


backtest_classes = (
    backtest_model.classes_
)


print(
    "\nModel classes:",
    backtest_classes
)


# GET HARD PREDICTIONS
backtest_predicted_results = (
    backtest_model.predict(
        X_backtest
    )
)


# CREATE PROBABILITY MAP
backtest_class_index = {
    class_name: index
    for index, class_name in enumerate(
        backtest_classes
    )
}


# EXTRACT INDIVIDUAL PROBABILITIES
backtest_home_probability = (
    backtest_probabilities[
        :,
        backtest_class_index["H"]
    ]
)

backtest_draw_probability = (
    backtest_probabilities[
        :,
        backtest_class_index["D"]
    ]
)

backtest_away_probability = (
    backtest_probabilities[
        :,
        backtest_class_index["A"]
    ]
)


# CREATE BACKTEST RESULTS DATAFRAME
backtest_results = backtest_df[
    [
        "season",
        "date",
        "round",
        "home_team",
        "away_team",
        "result"
    ]
].copy()


# ADD PREDICTION INFORMATION
backtest_results[
    "predicted_result"
] = backtest_predicted_results


backtest_results[
    "home_win_probability"
] = (
    backtest_home_probability * 100
)


backtest_results[
    "draw_probability"
] = (
    backtest_draw_probability * 100
)


backtest_results[
    "away_win_probability"
] = (
    backtest_away_probability * 100
)


# CONVERT RESULT CODE TO READABLE NAME
result_names = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}


backtest_results[
    "predicted_result_name"
] = (
    backtest_results[
        "predicted_result"
    ].map(result_names)
)


backtest_results[
    "correct"
] = (
    backtest_results[
        "predicted_result"
    ]
    ==
    backtest_results[
        "result"
    ]
)


# DISPLAY FIRST 10 BACKTEST PREDICTIONS
print(
    "\n===== FIRST 10 BACKTEST PREDICTIONS ====="
)

print(
    backtest_results.head(10)
)


# OVERALL ACCURACY
backtest_accuracy = accuracy_score(
    y_backtest,
    backtest_predicted_results
)


# OVERALL LOG LOSS
backtest_log_loss = log_loss(
    y_backtest,
    backtest_probabilities,
    labels=backtest_classes
)


# MULTICLASS BRIER SCORE
backtest_true_encoded = (
    y_backtest
    .map(backtest_class_index)
    .to_numpy()
)


backtest_brier_score = (
    (
        (
            backtest_probabilities -
            (
                backtest_true_encoded[:, None]
                ==
                range(
                    len(backtest_classes)
                )
            ).astype(float)
        ) ** 2
    )
    .sum(axis=1)
    .mean()
)


# NORMALIZED BRIER SCORE
backtest_normalized_brier = (
    backtest_brier_score /
    len(backtest_classes)
)


# DISPLAY OVERALL BACKTEST METRICS
print(
    "\n===== OVERALL BACKTEST PERFORMANCE ====="
)

print(
    "Accuracy:",
    round(
        backtest_accuracy,
        4
    )
)

print(
    "Log Loss:",
    round(
        backtest_log_loss,
        4
    )
)

print(
    "Brier Score:",
    round(
        backtest_brier_score,
        4
    )
)

print(
    "Normalized Brier Score:",
    round(
        backtest_normalized_brier,
        4
    )
)


# CONFUSION MATRIX
backtest_cm = confusion_matrix(
    y_backtest,
    backtest_predicted_results,
    labels=["A", "D", "H"]
)


print(
    "\n===== BACKTEST CONFUSION MATRIX ====="
)

print(
    "Classes: ['A', 'D', 'H']"
)

print(
    backtest_cm
)


# CLASSIFICATION REPORT
print(
    "\n===== BACKTEST CLASSIFICATION REPORT ====="
)

print(
    classification_report(
    y_backtest,
    backtest_predicted_results,
    labels=["A", "D", "H"],
    zero_division=0
)
)


# RESULT DISTRIBUTION
print(
    "\n===== BACKTEST ACTUAL RESULT DISTRIBUTION ====="
)

print(
    y_backtest.value_counts()
)


# PREDICTED RESULT DISTRIBUTION
print(
    "\n===== BACKTEST PREDICTED RESULT DISTRIBUTION ====="
)

print(
    pd.Series(
        backtest_predicted_results
    ).value_counts()
)


# SEASON-BY-SEASON BACKTEST
print(
    "\n===== BACKTEST BY SEASON ====="
)


backtest_season_metrics = []


for season in sorted(
    backtest_results["season"].unique()
):

    season_data = backtest_results[
        backtest_results["season"] == season
    ].copy()


    season_actual = (
        season_data["result"]
    )


    season_predicted = (
        season_data["predicted_result"]
    )


    season_probabilities = (
        season_data[
            [
                "away_win_probability",
                "draw_probability",
                "home_win_probability"
            ]
        ].to_numpy()
        / 100
    )


    season_accuracy = (
        accuracy_score(
            season_actual,
            season_predicted
        )
    )


    season_log_loss = (
        log_loss(
            season_actual,
            season_probabilities,
            labels=["A", "D", "H"]
        )
    )


    season_encoded = (
        season_actual
        .map({
            "A": 0,
            "D": 1,
            "H": 2
        })
        .to_numpy()
    )


    season_brier = (
        (
            (
                season_probabilities -
                (
                    season_encoded[:, None]
                    ==
                    range(3)
                ).astype(float)
            ) ** 2
        )
        .sum(axis=1)
        .mean()
    )


    backtest_season_metrics.append({

        "season":
            season,

        "matches":
            len(season_data),

        "accuracy":
            season_accuracy,

        "log_loss":
            season_log_loss,

        "brier_score":
            season_brier
    })


backtest_season_metrics_df = (
    pd.DataFrame(
        backtest_season_metrics
    )
)


print(
    backtest_season_metrics_df.round(4)
)


# CHECK PREDICTION PROBABILITY SUMS
probability_sum = (
    backtest_results[
        [
            "home_win_probability",
            "draw_probability",
            "away_win_probability"
        ]
    ].sum(axis=1)
)


print(
    "\n===== PROBABILITY VALIDATION ====="
)

print(
    "Minimum probability sum:",
    round(
        probability_sum.min(),
        6
    )
)

print(
    "Maximum probability sum:",
    round(
        probability_sum.max(),
        6
    )
)


if np.allclose(
    probability_sum,
    100,
    atol=0.01
):

    print(
        "Probability sum check: PASS"
    )

else:

    print(
        "Probability sum check: FAIL"
    )


# CHECK FOR MISSING BACKTEST VALUES
print(
    "\n===== BACKTEST MISSING VALUE CHECK ====="
)

print(
    backtest_results.isnull().sum()
)


# SAVE BACKTEST MATCH RESULTS
backtest_output = (
    BACKTEST_DIR /
    "champions_league_backtest_predictions_2023_24_to_2025_26.csv"
)


backtest_results.to_csv(
    backtest_output,
    index=False
)


print(
    "\nBacktest predictions saved successfully."
)

print(
    "Saved to:",
    backtest_output
)


# SAVE BACKTEST SUMMARY
backtest_summary = {

    "model":
        "Calibrated V2 CatBoost",

    "model_artifact":
        evaluation_model_path,

    "feature_version":
        "V2",

    "matches":
        len(backtest_results),

    "test_period": {
        "start":
            str(
                backtest_results[
                    "date"
                ].min()
            ),

        "end":
            str(
                backtest_results[
                    "date"
                ].max()
            )
    },

    "accuracy":
        float(
            backtest_accuracy
        ),

    "log_loss":
        float(
            backtest_log_loss
        ),

    "brier_score":
        float(
            backtest_brier_score
        ),

    "normalized_brier_score":
        float(
            backtest_normalized_brier
        )
}


backtest_summary_output = (
    BACKTEST_DIR /
    "champions_league_backtest_summary.json"
)


with open(
    backtest_summary_output,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        backtest_summary,
        file,
        indent=4
    )


print(
    "\nBacktest summary saved successfully."
)

print(
    "Saved to:",
    backtest_summary_output
)


print(
    "\n===== HISTORICAL BACKTEST COMPLETE ====="
)





# ERROR ANALYSIS
print(f"\n ERROR ANALYSIS")


# BASIC ERROR COUNTS
total_predictions = len(
    backtest_results
)

correct_predictions = (
    backtest_results["correct"]
    .sum()
)

incorrect_predictions = (
    total_predictions
    - correct_predictions
)

print(
    "\n===== BASIC ERROR SUMMARY ====="
)

print(
    "Total predictions:",
    total_predictions
)

print(
    "Correct predictions:",
    correct_predictions
)

print(
    "Incorrect predictions:",
    incorrect_predictions
)

print(
    "Accuracy:",
    round(
        correct_predictions / total_predictions,
        4
    )
)


# ACTUAL VS PREDICTED RESULT
print(
    "\n===== ACTUAL VS PREDICTED ====="
)

actual_vs_predicted = pd.crosstab(
    backtest_results["result"],
    backtest_results["predicted_result"],
    rownames=["Actual"],
    colnames=["Predicted"],
    dropna=False
)

print(
    actual_vs_predicted
)


# DRAW ANALYSIS
actual_draws = (
    backtest_results["result"] == "D"
).sum()

predicted_draws = (
    backtest_results["predicted_result"] == "D"
).sum()

correct_draws = (
    (
        backtest_results["result"] == "D"
    )
    &
    (
        backtest_results["predicted_result"] == "D"
    )
).sum()


print(
    "\n===== DRAW ANALYSIS ====="
)

print(
    "Actual draws:",
    actual_draws
)

print(
    "Predicted draws:",
    predicted_draws
)

print(
    "Correct draw predictions:",
    correct_draws
)


if actual_draws > 0:

    draw_recall = (
        correct_draws
        / actual_draws
    )

else:

    draw_recall = 0


print(
    "Draw recall:",
    round(
        draw_recall,
        4
    )
)


# PREDICTION CONFIDENCE
probability_columns = [
    "home_win_probability",
    "draw_probability",
    "away_win_probability"
]


backtest_results["prediction_confidence"] = (
    backtest_results[
        probability_columns
    ].max(axis=1)
)


backtest_results["prediction_margin"] = (
    backtest_results[
        probability_columns
    ].apply(
        lambda row:
            row.nlargest(2).iloc[0]
            -
            row.nlargest(2).iloc[1],
        axis=1
    )
)


# MOST CONFIDENT WRONG PREDICTIONS
most_confident_wrong = (
    backtest_results[
        backtest_results["correct"] == False
    ]
    .sort_values(
        "prediction_confidence",
        ascending=False
    )
)


print("\n===== MOST CONFIDENT WRONG PREDICTIONS =====")

most_confident_wrong_display = (
    most_confident_wrong[
        [
            "date",
            "season",
            "round",
            "home_team",
            "away_team",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "predicted_result",
            "result",
            "prediction_confidence",
            "prediction_margin"
        ]
    ]
    .head(20)
)

print(
    most_confident_wrong_display
)


# CONFIDENCE BUCKETS
backtest_results["confidence_bucket"] = pd.cut(
    backtest_results["prediction_confidence"],
    bins=[
        0,
        40,
        50,
        60,
        70,
        80,
        100.01
    ],
    labels=[
        "<40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80%+"
    ],
    right=False
)


confidence_analysis = (
    backtest_results
    .groupby(
        "confidence_bucket",
        observed=False
    )
    .agg(
        matches=("correct", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean")
    )
    .reset_index()
)


print(
    "\n===== CONFIDENCE ANALYSIS ====="
)

print(
    confidence_analysis.round(4)
)


# DRAW PROBABILITY ANALYSIS
backtest_results["draw_probability_bucket"] = pd.cut(
    backtest_results["draw_probability"],
    bins=[
        0,
        15,
        20,
        25,
        30,
        35,
        100.01
    ],
    labels=[
        "<15%",
        "15-20%",
        "20-25%",
        "25-30%",
        "30-35%",
        "35%+"
    ],
    right=False
)


draw_probability_analysis = (
    backtest_results
    .groupby(
        "draw_probability_bucket",
        observed=False
    )
    .agg(
        matches=("result", "count"),
        actual_draws=(
            "result",
            lambda x: (x == "D").sum()
        ),
        average_draw_probability=(
            "draw_probability",
            "mean"
        )
    )
    .reset_index()
)


draw_probability_analysis[
    "actual_draw_rate"
] = (
    draw_probability_analysis[
        "actual_draws"
    ]
    /
    draw_probability_analysis[
        "matches"
    ]
)


print(
    "\n===== DRAW PROBABILITY ANALYSIS ====="
)

print(
    draw_probability_analysis.round(4)
)


# ERROR TYPE
def classify_error(row):

    if row["correct"]:

        return "Correct"

    if (
        row["result"] == "D"
        and
        row["predicted_result"] != "D"
    ):

        return "Missed Draw"

    if (
        row["predicted_result"] == "D"
        and
        row["result"] != "D"
    ):

        return "False Draw"

    if (
        row["result"] == "H"
        and
        row["predicted_result"] == "A"
    ):

        return "Home vs Away"

    if (
        row["result"] == "A"
        and
        row["predicted_result"] == "H"
    ):

        return "Away vs Home"

    return "Other Error"


backtest_results["error_type"] = (
    backtest_results.apply(
        classify_error,
        axis=1
    )
)


print(
    "\n===== ERROR TYPE DISTRIBUTION ====="
)

print(
    backtest_results[
        "error_type"
    ].value_counts()
)


# ERROR ANALYSIS BY SEASON
season_error_analysis = (
    backtest_results
    .groupby("season")
    .agg(
        matches=("correct", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        missed_draws=(
            "error_type",
            lambda x: (x == "Missed Draw").sum()
        ),
        home_vs_away_errors=(
            "error_type",
            lambda x: (x == "Home vs Away").sum()
        ),
        away_vs_home_errors=(
            "error_type",
            lambda x: (x == "Away vs Home").sum()
        )
    )
    .reset_index()
)


print(
    "\n===== ERROR ANALYSIS BY SEASON ====="
)

print(
    season_error_analysis.round(4)
)


# ERROR ANALYSIS BY ROUND
round_error_analysis = (
    backtest_results
    .groupby("round")
    .agg(
        matches=("correct", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        missed_draws=(
            "error_type",
            lambda x: (x == "Missed Draw").sum()
        )
    )
    .reset_index()
)


print("\n===== ERROR ANALYSIS BY ROUND =====")

print(round_error_analysis.round(4))


# SAVE ERROR ANALYSIS RESULTS
BASE_DIR = Path(__file__).resolve().parent

ERROR_ANALYSIS_DIR = (
    BASE_DIR
    / "data"
    / "predictions"
)


error_analysis_output = (
    ERROR_ANALYSIS_DIR
    / "champions_league_error_analysis.csv"
)


backtest_results.to_csv(
    error_analysis_output,
    index=False
)


print(
    "\nError analysis predictions saved to:"
)

print(
    error_analysis_output
)


# SAVE SUMMARY
error_summary = {

    "total_predictions":
        int(total_predictions),

    "correct_predictions":
        int(correct_predictions),

    "incorrect_predictions":
        int(incorrect_predictions),

    "accuracy":
        float(
            correct_predictions
            /
            total_predictions
        ),

    "actual_draws":
        int(actual_draws),

    "predicted_draws":
        int(predicted_draws),

    "correct_draws":
        int(correct_draws),

    "draw_recall":
        float(draw_recall),

    "error_distribution":
        {
            str(key): int(value)
            for key, value
            in backtest_results[
                "error_type"
            ]
            .value_counts()
            .items()
        }

}


error_summary_output = (
    ERROR_ANALYSIS_DIR
    / "champions_league_error_analysis_summary.json"
)


with open(
    error_summary_output,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        error_summary,
        file,
        indent=4
    )


print("\nError analysis summary saved to:")

print(error_summary_output)

print("ERROR ANALYSIS COMPLETE")




print("\n ERROR ANALYSIS COMPLETE")

# CHECK BACKTEST RESULT COLUMNS
print("\n===== BACKTEST RESULT COLUMNS =====")
print(backtest_results.columns.tolist())


# MISSED DRAW ANALYSIS
print("\n" + "=" * 60)
print("MISSED DRAW ANALYSIS")
print("=" * 60)

# 1. Get all actual draws that were predicted incorrectly
missed_draws = backtest_results[
    (backtest_results["result"] == "D") &
    (backtest_results["predicted_result"] != "D")
].copy()

print("\n===== MISSED DRAW SUMMARY =====")
print(f"Actual draws missed: {len(missed_draws)}")

# 2. What did the model predict instead?
print("\n===== WHAT DID THE MODEL PREDICT? =====")

missed_prediction_counts = (
    missed_draws["predicted_result"]
    .value_counts()
    .rename_axis("predicted_result")
    .reset_index(name="matches")
)

print(missed_prediction_counts)

# 3. Probability statistics for the missed draws
print("\n===== MISSED DRAW PROBABILITY STATISTICS =====")

draw_probability_stats = pd.DataFrame({
    "statistic": [
        "Minimum",
        "25th Percentile",
        "Median",
        "Mean",
        "75th Percentile",
        "Maximum"
    ],
    "draw_probability": [
        missed_draws["draw_probability"].min(),
        missed_draws["draw_probability"].quantile(0.25),
        missed_draws["draw_probability"].median(),
        missed_draws["draw_probability"].mean(),
        missed_draws["draw_probability"].quantile(0.75),
        missed_draws["draw_probability"].max()
    ]
})

draw_probability_stats["draw_probability"] = (
    draw_probability_stats["draw_probability"].round(4)
)

print(draw_probability_stats)

# 4. Draw probability buckets
print("\n===== MISSED DRAWS BY DRAW PROBABILITY =====")

missed_draws["draw_probability_bucket"] = pd.cut(
    missed_draws["draw_probability"],
    bins=[0, 15, 20, 25, 30, 35, 100.01],
    labels=[
        "<15%",
        "15-20%",
        "20-25%",
        "25-30%",
        "30-35%",
        "35%+"
    ],
    right=False
)

missed_draw_probability = (
    missed_draws["draw_probability_bucket"]
    .value_counts()
    .sort_index()
    .rename_axis("draw_probability_bucket")
    .reset_index(name="missed_draws")
)

print(missed_draw_probability)

# 5. Confidence of missed draws
print("\n===== MISSED DRAW CONFIDENCE =====")

missed_draws["confidence_bucket"] = pd.cut(
    missed_draws["prediction_confidence"],
    bins=[0, 40, 50, 60, 70, 80, 100.01],
    labels=[
        "<40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80%+"
    ],
    right=False
)

missed_confidence = (
    missed_draws["confidence_bucket"]
    .value_counts()
    .sort_index()
    .rename_axis("confidence_bucket")
    .reset_index(name="missed_draws")
)

print(missed_confidence)

# 6. Missed draws by season
print("\n===== MISSED DRAWS BY SEASON =====")

missed_draws_by_season = (
    missed_draws
    .groupby("season", observed=False)
    .size()
    .reset_index(name="missed_draws")
)

print(missed_draws_by_season)

# 7. Missed draws by round
print("\n===== MISSED DRAWS BY ROUND =====")

missed_draws_by_round = (
    missed_draws
    .groupby("round", observed=False)
    .size()
    .reset_index(name="missed_draws")
    .sort_values("missed_draws", ascending=False)
)

print(missed_draws_by_round)

# 8. Display every missed draw
print("\n===== ALL 75 MISSED DRAWS =====")

missed_draw_display_columns = [
    "date",
    "season",
    "round",
    "home_team",
    "away_team",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "predicted_result",
    "prediction_confidence",
    "prediction_margin"
]

available_columns = [
    col for col in missed_draw_display_columns
    if col in missed_draws.columns
]

missed_draw_display = missed_draws[available_columns].copy()

for col in [
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "prediction_confidence",
    "prediction_margin"
]:
    if col in missed_draw_display.columns:
        missed_draw_display[col] = (
            missed_draw_display[col].round(2)
        )

print(missed_draw_display.to_string(index=False))


# 9. Save missed-draw analysis
MISSED_DRAW_ANALYSIS_FILE = (
    ERROR_ANALYSIS_DIR /
    "champions_league_missed_draw_analysis.csv"
)

missed_draws.to_csv(
    MISSED_DRAW_ANALYSIS_FILE,
    index=False
)

print(
    f"\nMissed draw analysis saved to:\n"
    f"{MISSED_DRAW_ANALYSIS_FILE}"
)

print("\nMISSED DRAW ANALYSIS COMPLETE")





# V3.5 DRAW DIAGNOSTIC ANALYSIS
print("\n" + "=" * 60)
print("V3.5 DRAW DIAGNOSTIC ANALYSIS")
print("=" * 60)

draw_diagnostic = backtest_results.copy()

# 1. ACTUAL DRAW RATE BY SEASON
print("\n===== ACTUAL DRAW RATE BY SEASON =====")

season_draw_analysis = (
    draw_diagnostic
    .groupby("season")
    .agg(
        matches=("result", "size"),
        actual_draws=("result", lambda x: (x == "D").sum())
    )
    .reset_index()
)

season_draw_analysis["draw_rate"] = (
    season_draw_analysis["actual_draws"] /
    season_draw_analysis["matches"] * 100
).round(2)

season_draw_analysis["missed_draws"] = (
    draw_diagnostic[draw_diagnostic["result"] == "D"]
    .groupby("season")
    .size()
    .reindex(season_draw_analysis["season"], fill_value=0)
    .values
)

season_draw_analysis["draw_recall"] = (
    season_draw_analysis["actual_draws"] -
    season_draw_analysis["missed_draws"]
)

season_draw_analysis["draw_recall"] = (
    season_draw_analysis["draw_recall"] /
    season_draw_analysis["actual_draws"] * 100
).round(2)

print(season_draw_analysis)


# 2. ACTUAL DRAW RATE BY ROUND
print("\n===== ACTUAL DRAW RATE BY ROUND =====")

round_draw_analysis = (
    draw_diagnostic
    .groupby("round")
    .agg(
        matches=("result", "size"),
        actual_draws=("result", lambda x: (x == "D").sum())
    )
    .reset_index()
)

round_draw_analysis["draw_rate"] = (
    round_draw_analysis["actual_draws"] /
    round_draw_analysis["matches"] * 100
).round(2)

print(
    round_draw_analysis
    .sort_values("draw_rate", ascending=False)
    .to_string(index=False)
)


# 3. DRAW RATE BY PREDICTED DRAW PROBABILITY
print("\n===== ACTUAL DRAW RATE BY PREDICTED DRAW PROBABILITY =====")

draw_diagnostic["draw_probability_bucket"] = pd.cut(
    draw_diagnostic["draw_probability"],
    bins=[0, 15, 20, 25, 30, 35, 40, 100.01],
    labels=[
        "<15%",
        "15-20%",
        "20-25%",
        "25-30%",
        "30-35%",
        "35-40%",
        "40%+"
    ],
    right=False
)

draw_probability_analysis = (
    draw_diagnostic
    .groupby("draw_probability_bucket", observed=False)
    .agg(
        matches=("result", "size"),
        actual_draws=("result", lambda x: (x == "D").sum())
    )
    .reset_index()
)

draw_probability_analysis["actual_draw_rate"] = (
    draw_probability_analysis["actual_draws"] /
    draw_probability_analysis["matches"] * 100
).round(2)

print(draw_probability_analysis)


# 4. DRAW RATE BY HOME/AWAY PROBABILITY GAP
print("\n===== DRAW RATE BY HOME/AWAY PROBABILITY GAP =====")

draw_diagnostic["home_away_probability_gap"] = (
    draw_diagnostic["home_win_probability"] -
    draw_diagnostic["away_win_probability"]
).abs()

draw_diagnostic["home_away_gap_bucket"] = pd.cut(
    draw_diagnostic["home_away_probability_gap"],
    bins=[0, 5, 10, 15, 20, 30, 40, 50, 100.01],
    labels=[
        "0-5%",
        "5-10%",
        "10-15%",
        "15-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50%+"
    ],
    right=False
)

home_away_gap_analysis = (
    draw_diagnostic
    .groupby("home_away_gap_bucket", observed=False)
    .agg(
        matches=("result", "size"),
        actual_draws=("result", lambda x: (x == "D").sum())
    )
    .reset_index()
)

home_away_gap_analysis["actual_draw_rate"] = (
    home_away_gap_analysis["actual_draws"] /
    home_away_gap_analysis["matches"] * 100
).round(2)

print(home_away_gap_analysis)


# 5. DRAW RATE BY PREDICTION MARGIN
print("\n===== DRAW RATE BY PREDICTION MARGIN =====")

draw_diagnostic["prediction_margin_bucket"] = pd.cut(
    draw_diagnostic["prediction_margin"],
    bins=[0, 5, 10, 15, 20, 30, 40, 50, 100.01],
    labels=[
        "0-5%",
        "5-10%",
        "10-15%",
        "15-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50%+"
    ],
    right=False
)

margin_analysis = (
    draw_diagnostic
    .groupby("prediction_margin_bucket", observed=False)
    .agg(
        matches=("result", "size"),
        actual_draws=("result", lambda x: (x == "D").sum())
    )
    .reset_index()
)

margin_analysis["actual_draw_rate"] = (
    margin_analysis["actual_draws"] /
    margin_analysis["matches"] * 100
).round(2)

print(margin_analysis)


# 6. BALANCED MATCHES
print("\n===== BALANCED MATCH ANALYSIS =====")

balanced_matches = draw_diagnostic[
    draw_diagnostic["home_away_probability_gap"] <= 10
].copy()

balanced_total = len(balanced_matches)
balanced_draws = (balanced_matches["result"] == "D").sum()

if balanced_total > 0:
    balanced_draw_rate = balanced_draws / balanced_total * 100
else:
    balanced_draw_rate = 0

print(f"Balanced matches (Home/Away gap <= 10%): {balanced_total}")
print(f"Actual draws among balanced matches: {balanced_draws}")
print(f"Draw rate: {balanced_draw_rate:.2f}%")


# 7. MISSED DRAWS VS CORRECT NON-DRAW PREDICTIONS
print("\n===== MISSED DRAWS VS CORRECT NON-DRAW PREDICTIONS =====")

missed_draws = draw_diagnostic[
    (draw_diagnostic["result"] == "D") &
    (draw_diagnostic["predicted_result"] != "D")
].copy()

correct_non_draws = draw_diagnostic[
    (draw_diagnostic["result"] != "D") &
    (draw_diagnostic["predicted_result"] == draw_diagnostic["result"])
].copy()

print(f"Missed draws: {len(missed_draws)}")
print(f"Correct non-draw predictions: {len(correct_non_draws)}")


# 8. COMPARE EXISTING MODEL FEATURES
v2_features = [
    "home_goal_scoring_edge",
    "home_defensive_edge",
    "home_attack_vs_away_defense",
    "away_attack_vs_home_defense",
    "win_rate_edge",
    "venue_win_rate_edge",
    "recent_points_edge",
    "experience_edge",
    "days_since_match_edge"
]

available_v2_features = [
    feature for feature in v2_features
    if feature in draw_diagnostic.columns
]

print("\n===== FEATURE COMPARISON =====")

if available_v2_features:

    missed_draw_feature_means = (
        missed_draws[available_v2_features]
        .mean()
        .rename("missed_draw_mean")
    )

    correct_non_draw_feature_means = (
        correct_non_draws[available_v2_features]
        .mean()
        .rename("correct_non_draw_mean")
    )

    feature_comparison = pd.concat(
        [
            missed_draw_feature_means,
            correct_non_draw_feature_means
        ],
        axis=1
    )

    feature_comparison["difference"] = (
        feature_comparison["missed_draw_mean"] -
        feature_comparison["correct_non_draw_mean"]
    )

    feature_comparison = feature_comparison.round(4)

    print(feature_comparison)

else:
    print("No V2 feature columns found in backtest_results.")


# 9. LEAGUE PHASE VS GROUP STAGE
print("\n===== LEAGUE PHASE VS GROUP STAGE =====")

phase_comparison = draw_diagnostic[
    draw_diagnostic["round"].isin([
        "League phase",
        "Group stage"
    ])
].copy()

phase_analysis = (
    phase_comparison
    .groupby("round")
    .agg(
        matches=("result", "size"),
        actual_draws=("result", lambda x: (x == "D").sum())
    )
    .reset_index()
)

phase_analysis["draw_rate"] = (
    phase_analysis["actual_draws"] /
    phase_analysis["matches"] * 100
).round(2)

print(phase_analysis)


# 10. HIGH DRAW-PROBABILITY MATCHES
print("\n===== HIGHEST DRAW PROBABILITY MATCHES =====")

highest_draw_probability = (
    draw_diagnostic
    .sort_values("draw_probability", ascending=False)
    [
        [
            "date",
            "season",
            "round",
            "home_team",
            "away_team",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "result",
            "predicted_result"
        ]
    ]
    .head(20)
)

print(highest_draw_probability.to_string(index=False))


# 11. SAVE V3.5 DIAGNOSTIC RESULTS
DRAW_DIAGNOSTIC_FILE = (
    ERROR_ANALYSIS_DIR /
    "champions_league_v3_5_draw_diagnostics.csv"
)

draw_diagnostic.to_csv(
    DRAW_DIAGNOSTIC_FILE,
    index=False
)

print(
    f"\nV3.5 diagnostic data saved to:\n"
    f"{DRAW_DIAGNOSTIC_FILE}"
)

print("\nV3.5 DRAW DIAGNOSTIC ANALYSIS COMPLETE")





# V3.6 FEATURE-LEVEL DRAW ANALYSIS
print("\n" + "=" * 60)
print("V3.6 FEATURE-LEVEL DRAW ANALYSIS")
print("=" * 60)

# 1. LOAD V2 FEATURE DATA
FEATURES_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "champions_league_features_v2.csv"
)

v2_features_df = pd.read_csv(FEATURES_FILE)

print("\n===== V2 FEATURE DATA =====")
print(f"Rows: {len(v2_features_df)}")
print(f"Columns: {len(v2_features_df.columns)}")


# 2. CREATE MATCH KEY
def create_match_key(df):
    return (
        df["date"].astype(str) + "|" +
        df["home_team"].astype(str) + "|" +
        df["away_team"].astype(str)
    )


draw_diagnostic["match_key"] = create_match_key(draw_diagnostic)
v2_features_df["match_key"] = create_match_key(v2_features_df)


# 3. V2 FEATURES WE WANT TO STUDY
v2_feature_columns = [
    "home_goal_scoring_edge",
    "home_defensive_edge",
    "home_attack_vs_away_defense",
    "away_attack_vs_home_defense",
    "win_rate_edge",
    "venue_win_rate_edge",
    "recent_points_edge",
    "experience_edge",
    "days_since_match_edge"
]

available_features = [
    feature
    for feature in v2_feature_columns
    if feature in v2_features_df.columns
]

print("\n===== AVAILABLE V2 FEATURES =====")
print(available_features)


# 4. MERGE BACKTEST RESULTS WITH V2 FEATURES
feature_analysis = draw_diagnostic.merge(
    v2_features_df[
        ["match_key"] + available_features
    ],
    on="match_key",
    how="left",
    suffixes=("", "_feature")
)

print("\n===== FEATURE MERGE CHECK =====")

print(
    f"Backtest matches: {len(draw_diagnostic)}"
)

print(
    f"Matches after merge: {len(feature_analysis)}"
)

missing_feature_rows = (
    feature_analysis[available_features]
    .isna()
    .all(axis=1)
    .sum()
)

print(
    f"Matches with no V2 feature match: "
    f"{missing_feature_rows}"
)


# 5. ACTUAL OUTCOME GROUPS
actual_draws = feature_analysis[
    feature_analysis["result"] == "D"
].copy()

actual_home_wins = feature_analysis[
    feature_analysis["result"] == "H"
].copy()

actual_away_wins = feature_analysis[
    feature_analysis["result"] == "A"
].copy()

print("\n===== ACTUAL OUTCOME COUNTS =====")

print(
    f"Actual Draws: {len(actual_draws)}"
)

print(
    f"Actual Home Wins: {len(actual_home_wins)}"
)

print(
    f"Actual Away Wins: {len(actual_away_wins)}"
)


# 6. FEATURE MEANS BY ACTUAL RESULT
print("\n===== FEATURE MEANS BY ACTUAL RESULT =====")

feature_means = pd.DataFrame({
    "Draw": actual_draws[available_features].mean(),
    "Home": actual_home_wins[available_features].mean(),
    "Away": actual_away_wins[available_features].mean()
})

feature_means = feature_means.round(4)

print(feature_means)


# 7. FEATURE MEDIANS BY ACTUAL RESULT
print("\n===== FEATURE MEDIANS BY ACTUAL RESULT =====")

feature_medians = pd.DataFrame({
    "Draw": actual_draws[available_features].median(),
    "Home": actual_home_wins[available_features].median(),
    "Away": actual_away_wins[available_features].median()
})

feature_medians = feature_medians.round(4)

print(feature_medians)


# 8. DRAW VS NON-DRAW COMPARISON
print("\n===== DRAW VS NON-DRAW FEATURE COMPARISON =====")

non_draws = feature_analysis[
    feature_analysis["result"] != "D"
].copy()

draw_vs_non_draw = pd.DataFrame({
    "draw_mean": actual_draws[available_features].mean(),
    "non_draw_mean": non_draws[available_features].mean()
})

draw_vs_non_draw["difference"] = (
    draw_vs_non_draw["draw_mean"] -
    draw_vs_non_draw["non_draw_mean"]
)

draw_vs_non_draw["absolute_difference"] = (
    draw_vs_non_draw["difference"].abs()
)

draw_vs_non_draw = (
    draw_vs_non_draw
    .sort_values(
        "absolute_difference",
        ascending=False
    )
    .round(4)
)

print(draw_vs_non_draw)


# 9. FEATURE SPREAD
print("\n===== DRAW FEATURE SPREAD =====")

draw_feature_spread = pd.DataFrame({
    "mean": actual_draws[available_features].mean(),
    "median": actual_draws[available_features].median(),
    "std": actual_draws[available_features].std(),
    "minimum": actual_draws[available_features].min(),
    "maximum": actual_draws[available_features].max()
})

draw_feature_spread = draw_feature_spread.round(4)

print(draw_feature_spread)


# 10. MISSED DRAWS FEATURE PROFILE
print("\n===== MISSED DRAW FEATURE PROFILE =====")

missed_draw_feature_data = feature_analysis[
    (feature_analysis["result"] == "D") &
    (feature_analysis["predicted_result"] != "D")
].copy()

missed_draw_feature_profile = pd.DataFrame({
    "missed_draw_mean": (
        missed_draw_feature_data[available_features].mean()
    ),
    "all_draw_mean": (
        actual_draws[available_features].mean()
    ),
    "non_draw_mean": (
        non_draws[available_features].mean()
    )
})

missed_draw_feature_profile["missed_vs_all_draws"] = (
    missed_draw_feature_profile["missed_draw_mean"] -
    missed_draw_feature_profile["all_draw_mean"]
)

missed_draw_feature_profile["missed_vs_non_draw"] = (
    missed_draw_feature_profile["missed_draw_mean"] -
    missed_draw_feature_profile["non_draw_mean"]
)

missed_draw_feature_profile = (
    missed_draw_feature_profile.round(4)
)

print(missed_draw_feature_profile)


# 11. FEATURE RANGE FOR MISSED DRAWS
print("\n===== MISSED DRAW FEATURE RANGES =====")

missed_draw_feature_ranges = pd.DataFrame({
    "minimum": (
        missed_draw_feature_data[available_features].min()
    ),
    "25th_percentile": (
        missed_draw_feature_data[available_features]
        .quantile(0.25)
    ),
    "median": (
        missed_draw_feature_data[available_features]
        .median()
    ),
    "75th_percentile": (
        missed_draw_feature_data[available_features]
        .quantile(0.75)
    ),
    "maximum": (
        missed_draw_feature_data[available_features].max()
    )
})

missed_draw_feature_ranges = (
    missed_draw_feature_ranges.round(4)
)

print(missed_draw_feature_ranges)


# 12. FEATURE ANALYSIS BY ROUND
print("\n===== MISSED DRAW FEATURE ANALYSIS BY ROUND =====")

missed_draw_round_features = (
    missed_draw_feature_data
    .groupby("round")[available_features]
    .mean()
    .round(4)
)

print(missed_draw_round_features)


# 13. SAVE FEATURE ANALYSIS
V36_FEATURE_ANALYSIS_FILE = (
    ERROR_ANALYSIS_DIR /
    "champions_league_v3_6_feature_draw_analysis.csv"
)

feature_analysis.to_csv(
    V36_FEATURE_ANALYSIS_FILE,
    index=False
)

print(
    f"\nV3.6 feature analysis saved to:\n"
    f"{V36_FEATURE_ANALYSIS_FILE}"
)

print("\nV3.6 FEATURE-LEVEL DRAW ANALYSIS COMPLETE")





# V4-A: CURRENT-SEASON FEATURES
print(f"\n # V4-A: CURRENT-SEASON FEATURES")

def add_current_season_features(df):
    """
    Add current Champions League season statistics.

    Features are calculated strictly BEFORE each match.

    The input dataframe must contain:
        season
        date
        home_team
        away_team
        home_goals
        away_goals
    """

    df = df.copy()


    # Check required columns
    required_columns = [
        "season",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


    # Prepare data
    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values(
            ["date", "home_team", "away_team"]
        )
        .reset_index(drop=True)
    )


    # Feature columns
    feature_columns = [
        "home_current_season_matches",
        "away_current_season_matches",

        "home_current_season_win_rate",
        "away_current_season_win_rate",

        "home_current_season_draw_rate",
        "away_current_season_draw_rate",

        "home_current_season_goal_difference",
        "away_current_season_goal_difference",

        "home_current_season_avg_goals_for",
        "away_current_season_avg_goals_for",

        "home_current_season_avg_goals_against",
        "away_current_season_avg_goals_against",

        "current_season_win_rate_edge",
        "current_season_draw_rate_edge",
        "current_season_goal_difference_edge",
        "current_season_attack_edge",
        "current_season_defense_edge",
    ]

    for column in feature_columns:
        df[column] = 0.0

    # Store statistics for each team in each season
    team_stats = {}

    # Process matches chronologically
    for index, row in df.iterrows():

        season = row["season"]

        home_team = row["home_team"]
        away_team = row["away_team"]

        home_key = (season, home_team)
        away_key = (season, away_team)

        # Get statistics BEFORE current match
        home_stats = team_stats.get(
            home_key,
            {
                "matches": 0,
                "wins": 0,
                "draws": 0,
                "goals_for": 0,
                "goals_against": 0,
            },
        )

        away_stats = team_stats.get(
            away_key,
            {
                "matches": 0,
                "wins": 0,
                "draws": 0,
                "goals_for": 0,
                "goals_against": 0,
            },
        )

        # HOME TEAM
        home_matches = home_stats["matches"]

        if home_matches > 0:

            home_win_rate = (
                home_stats["wins"] / home_matches
            )

            home_draw_rate = (
                home_stats["draws"] / home_matches
            )

            home_avg_goals_for = (
                home_stats["goals_for"] / home_matches
            )

            home_avg_goals_against = (
                home_stats["goals_against"] / home_matches
            )

        else:

            home_win_rate = 0.0
            home_draw_rate = 0.0
            home_avg_goals_for = 0.0
            home_avg_goals_against = 0.0

        home_goal_difference = (
            home_stats["goals_for"]
            - home_stats["goals_against"]
        )

        # AWAY TEAM
        away_matches = away_stats["matches"]

        if away_matches > 0:

            away_win_rate = (
                away_stats["wins"] / away_matches
            )

            away_draw_rate = (
                away_stats["draws"] / away_matches
            )

            away_avg_goals_for = (
                away_stats["goals_for"] / away_matches
            )

            away_avg_goals_against = (
                away_stats["goals_against"] / away_matches
            )

        else:

            away_win_rate = 0.0
            away_draw_rate = 0.0
            away_avg_goals_for = 0.0
            away_avg_goals_against = 0.0

        away_goal_difference = (
            away_stats["goals_for"]
            - away_stats["goals_against"]
        )

        # SAVE CURRENT-SEASON FEATURES
        df.loc[
            index, "home_current_season_matches"
        ] = home_matches

        df.loc[
            index, "away_current_season_matches"
        ] = away_matches

        df.loc[
            index, "home_current_season_win_rate"
        ] = home_win_rate

        df.loc[
            index, "away_current_season_win_rate"
        ] = away_win_rate

        df.loc[
            index, "home_current_season_draw_rate"
        ] = home_draw_rate

        df.loc[
            index, "away_current_season_draw_rate"
        ] = away_draw_rate

        df.loc[
            index, "home_current_season_goal_difference"
        ] = home_goal_difference

        df.loc[
            index, "away_current_season_goal_difference"
        ] = away_goal_difference

        df.loc[
            index, "home_current_season_avg_goals_for"
        ] = home_avg_goals_for

        df.loc[
            index, "away_current_season_avg_goals_for"
        ] = away_avg_goals_for

        df.loc[
            index, "home_current_season_avg_goals_against"
        ] = home_avg_goals_against

        df.loc[
            index, "away_current_season_avg_goals_against"
        ] = away_avg_goals_against

        # RELATIVE FEATURES
        df.loc[
            index, "current_season_win_rate_edge"
        ] = home_win_rate - away_win_rate

        df.loc[
            index, "current_season_draw_rate_edge"
        ] = home_draw_rate - away_draw_rate

        df.loc[
            index, "current_season_goal_difference_edge"
        ] = (
            home_goal_difference
            - away_goal_difference
        )

        df.loc[
            index, "current_season_attack_edge"
        ] = (
            home_avg_goals_for
            - away_avg_goals_for
        )

        # Lower goals conceded = better defense
        df.loc[
            index, "current_season_defense_edge"
        ] = (
            away_avg_goals_against
            - home_avg_goals_against
        )

        # UPDATE STATISTICS AFTER CURRENT MATCH
        # This is deliberately AFTER feature creation.
        # Therefore the current match cannot leak into itself.

        home_goals = row["home_goals"]
        away_goals = row["away_goals"]

        # HOME TEAM UPDATE

        home_stats["matches"] += 1

        home_stats["goals_for"] += home_goals

        home_stats["goals_against"] += away_goals

        if home_goals > away_goals:

            home_stats["wins"] += 1

        elif home_goals == away_goals:

            home_stats["draws"] += 1

        # AWAY TEAM UPDATE

        away_stats["matches"] += 1

        away_stats["goals_for"] += away_goals

        away_stats["goals_against"] += home_goals

        if away_goals > home_goals:

            away_stats["wins"] += 1

        elif away_goals == home_goals:

            away_stats["draws"] += 1

        # Save updated statistics
        team_stats[home_key] = home_stats
        team_stats[away_key] = away_stats

    return df

print("\nFEATURES_V2 COLUMNS:")
print(features_v2.columns.tolist())

print("\nCLEAN DATAFRAME COLUMNS:")
print(clean_df.columns.tolist())


# BUILD V4-A DATASET FROM FULL CLEAN MASTER DATA
clean_master_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "champions_league_all_seasons_clean.csv"
)

clean_master_df = pd.read_csv(clean_master_path)

print("\nCLEAN MASTER DATASET:")
print("Shape:", clean_master_df.shape)

print("\nSEASONS:")
print(
    clean_master_df["season"]
    .value_counts()
    .sort_index()
)

# Build V4-A features
features_v4a = add_current_season_features(clean_master_df)

print("\nACTUAL V4-A COLUMNS:")
print(features_v4a.columns.tolist())

print("\nV4-A FEATURE ENGINEERING")
print("Shape:", features_v4a.shape)

print("\nV4-A SEASONS:")
print(
    features_v4a["season"]
    .value_counts()
    .sort_index()
)

print("\n")

# V4-A FEATURE LIST
v4a_features = [
    "home_current_season_matches",
    "away_current_season_matches",

    "home_current_season_win_rate",
    "away_current_season_win_rate",

    "home_current_season_draw_rate",
    "away_current_season_draw_rate",

    "home_current_season_goal_difference",
    "away_current_season_goal_difference",

    "home_current_season_avg_goals_for",
    "away_current_season_avg_goals_for",

    "home_current_season_avg_goals_against",
    "away_current_season_avg_goals_against",

    "current_season_win_rate_edge",
    "current_season_draw_rate_edge",
    "current_season_goal_difference_edge",

    "current_season_attack_edge",
    "current_season_defense_edge",
]


print("\nV4-A FEATURES:")
print(v4a_features)

print("\nACTUAL V4-A COLUMNS:")
print(features_v4a.columns.tolist())

# V4-A SANITY AND LEAKAGE CHECK
print("\n" + "=" * 60)
print("V4-A SANITY AND LEAKAGE CHECK")
print("=" * 60)


# 1. BASIC SHAPE CHECK
print("\n1. BASIC CHECK")

print("Rows:", len(features_v4a))
print("Columns:", len(features_v4a.columns))

assert len(features_v4a) == 2122, (
    f"Expected 2122 rows, got {len(features_v4a)}"
)

print("✓ Row count correct")


# 2. REQUIRED COLUMNS CHECK
required_v4a_columns = [
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
] + v4a_features

missing_v4a_columns = [
    column
    for column in required_v4a_columns
    if column not in features_v4a.columns
]

print("\n2. REQUIRED COLUMNS")

if missing_v4a_columns:
    print("Missing columns:", missing_v4a_columns)
    raise ValueError(
        f"V4-A missing columns: {missing_v4a_columns}"
    )

print("✓ All required columns present")


# 3. MISSING-VALUE CHECK
print("\n3. MISSING VALUES")

missing_values = (
    features_v4a[v4a_features]
    .isna()
    .sum()
)

print(missing_values)

assert missing_values.sum() == 0, (
    "V4-A contains missing feature values"
)

print("✓ No missing V4-A feature values")


# 4. DUPLICATE MATCH CHECK
print("\n4. DUPLICATE MATCH CHECK")

duplicate_matches = features_v4a.duplicated(
    subset=["season", "date", "home_team", "away_team"]
).sum()

print("Duplicate matches:", duplicate_matches)

assert duplicate_matches == 0, (
    f"Found {duplicate_matches} duplicate matches"
)

print("✓ No duplicate matches")


# 5. SEASON COUNT CHECK
print("\n5. SEASON COUNTS")

season_counts = (
    features_v4a["season"]
    .value_counts()
    .sort_index()
)

print(season_counts)

assert len(season_counts) == 16, (
    f"Expected 16 seasons, got {len(season_counts)}"
)

print("✓ All 16 seasons present")


# 6. FIRST-MATCH CURRENT-SEASON CHECK
print("\n6. FIRST-MATCH CHECK")

first_matches = (
    features_v4a
    .sort_values(["season", "date"])
    .groupby("season")
    .head(1)
)

first_match_columns = [
    "season",
    "date",
    "home_team",
    "away_team",
    "home_current_season_matches",
    "away_current_season_matches",
]

print(
    first_matches[first_match_columns].to_string(
        index=False
    )
)

first_match_counts = first_matches[
    [
        "home_current_season_matches",
        "away_current_season_matches",
    ]
]

assert (
    first_match_counts == 0
).all().all(), (
    "First matches contain prior current-season matches"
)

print("✓ First matches correctly start at zero")


# 7. NO NEGATIVE CURRENT-SEASON MATCH COUNTS
print("\n7. NEGATIVE-VALUE CHECK")

count_columns = [
    "home_current_season_matches",
    "away_current_season_matches",
]

negative_counts = (
    features_v4a[count_columns] < 0
).sum()

print(negative_counts)

assert negative_counts.sum() == 0, (
    "Negative current-season match counts detected"
)

print("✓ No negative match counts")


# 8. CURRENT-SEASON MATCH COUNT SANITY
print("\n8. MATCH COUNT RANGE")

for column in count_columns:

    print(
        f"{column}: "
        f"min={features_v4a[column].min()}, "
        f"max={features_v4a[column].max()}"
    )

assert (
    features_v4a[count_columns] >= 0
).all().all()

print("✓ Match-count ranges valid")


# 9. GOAL-DIFFERENCE CONSISTENCY
print("\n9. GOAL DIFFERENCE CHECK")

home_expected_goal_difference = (
    features_v4a["home_current_season_avg_goals_for"]
    * features_v4a["home_current_season_matches"]
    -
    features_v4a["home_current_season_avg_goals_against"]
    * features_v4a["home_current_season_matches"]
)

away_expected_goal_difference = (
    features_v4a["away_current_season_avg_goals_for"]
    * features_v4a["away_current_season_matches"]
    -
    features_v4a["away_current_season_avg_goals_against"]
    * features_v4a["away_current_season_matches"]
)

home_goal_difference_check = np.isclose(
    features_v4a["home_current_season_goal_difference"],
    home_expected_goal_difference,
    atol=1e-10
)

away_goal_difference_check = np.isclose(
    features_v4a["away_current_season_goal_difference"],
    away_expected_goal_difference,
    atol=1e-10
)

print(
    "Home goal difference mismatches:",
    (~home_goal_difference_check).sum()
)

print(
    "Away goal difference mismatches:",
    (~away_goal_difference_check).sum()
)

assert home_goal_difference_check.all(), (
    "Home goal difference calculation has mismatches"
)

assert away_goal_difference_check.all(), (
    "Away goal difference calculation has mismatches"
)

print("✓ Goal-difference calculations are consistent")


# 10. V4-A FEATURE SUMMARY
print("\n10. V4-A FEATURE SUMMARY")

print(
    features_v4a[v4a_features]
    .describe()
    .T[
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    ]
)


# FINAL RESULT
print("\n" + "=" * 60)
print("V4-A SANITY CHECK COMPLETE")
print("=" * 60)





# V4-A + V2 COMBINED DATASET
print("\n" + "=" * 60)
print("V4-A + V2 COMBINED DATASET")
print("=" * 60)


# Keep only V4-A features and matching keys
v4a_features_only = features_v4a[
    [
        "season",
        "date",
        "home_team",
        "away_team",
    ] + v4a_features
].copy()


# Merge V4-A features into the existing V2 dataset
features_v4a_combined = features_v2.merge(
    v4a_features_only,
    on=[
        "season",
        "date",
        "home_team",
        "away_team",
    ],
    how="left",
    validate="one_to_one",
)


# BASIC CHECKS
print("\nCOMBINED DATASET SHAPE:")
print(features_v4a_combined.shape)


# Check row count
assert len(features_v4a_combined) == 2122, (
    f"Expected 2122 rows, got {len(features_v4a_combined)}"
)

print("✓ Row count correct")


# Check V4-A missing values
combined_missing = (
    features_v4a_combined[v4a_features]
    .isna()
    .sum()
)

print("\nV4-A MISSING VALUES:")
print(combined_missing)

assert combined_missing.sum() == 0, (
    "Missing V4-A values found after merge"
)

print("✓ No missing V4-A values")


# Check duplicates
combined_duplicates = features_v4a_combined.duplicated(
    subset=[
        "season",
        "date",
        "home_team",
        "away_team",
    ]
).sum()

print("\nDUPLICATE MATCHES:", combined_duplicates)

assert combined_duplicates == 0, (
    f"Found {combined_duplicates} duplicate matches"
)

print("✓ No duplicate matches")


# Check result consistency
result_mismatches = (
    features_v4a_combined["result"]
    != features_v4a_combined["result"]
).sum()

print("\nRESULT MISMATCHES:", result_mismatches)


# Save combined dataset
v4a_combined_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "champions_league_features_v4a_combined.csv"
)

features_v4a_combined.to_csv(
    v4a_combined_path,
    index=False
)

print(
    f"\n✓ V4-A combined dataset saved to:\n"
    f"{v4a_combined_path}"
)

print("\n" + "=" * 60)
print("V4-A + V2 COMBINATION COMPLETE")
print("=" * 60)




print("\n V4-A MODEL FEATURE LIST ")
# V4-A MODEL FEATURE LIST
# Existing V2 model features
V2_MODEL_FEATURES = [
    "home_matches_before",
    "away_matches_before",
    "home_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "home_win_rate",
    "away_win_rate",
    "home_home_win_rate",
    "away_away_win_rate",
    "home_recent_points_5",
    "away_recent_points_5",
    "home_days_since_match_capped",
    "away_days_since_match_capped",
    "home_has_previous_match",
    "away_has_previous_match",
    "home_goal_scoring_edge",
    "home_defensive_edge",
    "home_attack_vs_away_defense",
    "away_attack_vs_home_defense",
    "win_rate_edge",
    "venue_win_rate_edge",
    "recent_points_edge",
    "experience_edge",
    "days_since_match_edge",
]


# Add V4-A features
V4A_MODEL_FEATURES = (
    V2_MODEL_FEATURES
    + v4a_features
)


print("\n" + "=" * 60)
print("V4-A MODEL FEATURES")
print("=" * 60)

print("V2 features:", len(V2_MODEL_FEATURES))
print("V4-A features:", len(v4a_features))
print("Combined features:", len(V4A_MODEL_FEATURES))


# V4-A FEATURE AVAILABILITY CHECK
missing_model_features = [
    feature
    for feature in V4A_MODEL_FEATURES
    if feature not in features_v4a_combined.columns
]

print("\nMISSING MODEL FEATURES:")
print(missing_model_features)

assert len(missing_model_features) == 0, (
    f"Missing model features: {missing_model_features}"
)

print("✓ All 42 model features are available")



# SAME TIME-AWARE TRAIN / TEST SPLIT
V4A_TEST_START_DATE = pd.Timestamp("2023-11-29")

v4a_train_df = features_v4a_combined[
    features_v4a_combined["date"] < V4A_TEST_START_DATE
].copy()

v4a_test_df = features_v4a_combined[
    features_v4a_combined["date"] >= V4A_TEST_START_DATE
].copy()

print("\n" + "=" * 60)
print("V4-A TIME-AWARE SPLIT")
print("=" * 60)

print("Training rows:", len(v4a_train_df))
print("Test rows:", len(v4a_test_df))

print(
    "Train dates:",
    v4a_train_df["date"].min(),
    "to",
    v4a_train_df["date"].max()
)

print(
    "Test dates:",
    v4a_test_df["date"].min(),
    "to",
    v4a_test_df["date"].max()
)

assert len(v4a_train_df) == 1691
assert len(v4a_test_df) == 431

assert (
    v4a_train_df["date"].max()
    <
    v4a_test_df["date"].min()
)

print("✓ Same 431-match holdout confirmed")
print("✓ No train/test date overlap")



# PREPARE V4-A TRAINING DATA
X_v4a_train = v4a_train_df[V4A_MODEL_FEATURES]
y_v4a_train = v4a_train_df["result"]

X_v4a_test = v4a_test_df[V4A_MODEL_FEATURES]
y_v4a_test = v4a_test_df["result"]

print("\nV4-A TRAINING SHAPE:", X_v4a_train.shape)
print("V4-A TEST SHAPE:", X_v4a_test.shape)

print("\nTARGET DISTRIBUTION - TRAIN:")
print(y_v4a_train.value_counts())

print("\nTARGET DISTRIBUTION - TEST:")
print(y_v4a_test.value_counts())



print("\n TRAIN V4-A CATBOOST ")

# TRAIN V4-A CATBOOST
v4a_catboost = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="MultiClass",
    verbose=False
)


# TIME-AWARE CALIBRATION
v4a_calibrated_model = CalibratedClassifierCV(
    v4a_catboost,
    cv=TimeSeriesSplit(n_splits=5),
    method="sigmoid",
    ensemble=True
)


# FIT MODEL
v4a_calibrated_model.fit(
    X_v4a_train,
    y_v4a_train
)

print("✓ V4-A CatBoost training complete")
print("✓ V4-A calibration complete")


# V4-A TEST PREDICTIONS
v4a_test_probabilities = v4a_calibrated_model.predict_proba(
    X_v4a_test
)

v4a_test_predictions = v4a_calibrated_model.predict(
    X_v4a_test
)

print("Probability shape:", v4a_test_probabilities.shape)
print("Prediction count:", len(v4a_test_predictions))

print("\nMODEL CLASSES:")
print(v4a_calibrated_model.classes_)



# V4-A EVALUATION
v4a_accuracy = accuracy_score(
    y_v4a_test,
    v4a_test_predictions
)

v4a_logloss = log_loss(
    y_v4a_test,
    v4a_test_probabilities,
    labels=v4a_calibrated_model.classes_
)

# Convert actual labels to one-hot
class_order = list(v4a_calibrated_model.classes_)

y_test_encoded = np.array([
    [1 if actual == cls else 0 for cls in class_order]
    for actual in y_v4a_test
])

# Multiclass Brier score
v4a_brier = np.mean(
    np.sum(
        (y_test_encoded - v4a_test_probabilities) ** 2,
        axis=1
    )
)

v4a_normalized_brier = v4a_brier / 3

print("\n" + "=" * 60)
print("V4-A CALIBRATED CATBOOST RESULTS")
print("=" * 60)

print(f"Accuracy:          {v4a_accuracy:.4f}")
print(f"Log Loss:          {v4a_logloss:.4f}")
print(f"Brier Score:       {v4a_brier:.4f}")
print(f"Normalized Brier:  {v4a_normalized_brier:.4f}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(
    y_v4a_test,
    v4a_test_predictions,
    labels=["A", "D", "H"]
))





# V4-A PROBABILITY DISTRIBUTION ANALYSIS
# Probability column positions
class_order = list(v4a_calibrated_model.classes_)

a_idx = class_order.index("A")
d_idx = class_order.index("D")
h_idx = class_order.index("H")

v4a_away_prob = v4a_test_probabilities[:, a_idx]
v4a_draw_prob = v4a_test_probabilities[:, d_idx]
v4a_home_prob = v4a_test_probabilities[:, h_idx]

print("\n" + "=" * 60)
print("V4-A PROBABILITY DISTRIBUTION")
print("=" * 60)

print(f"Average Away probability:  {v4a_away_prob.mean():.4f}")
print(f"Average Draw probability:  {v4a_draw_prob.mean():.4f}")
print(f"Average Home probability:  {v4a_home_prob.mean():.4f}")

print("\nProbability ranges:")

print(
    f"Away: min={v4a_away_prob.min():.4f}, "
    f"max={v4a_away_prob.max():.4f}"
)

print(
    f"Draw: min={v4a_draw_prob.min():.4f}, "
    f"max={v4a_draw_prob.max():.4f}"
)

print(
    f"Home: min={v4a_home_prob.min():.4f}, "
    f"max={v4a_home_prob.max():.4f}"
)



# ACTUAL VS PREDICTED RESULT DISTRIBUTION
actual_distribution = y_v4a_test.value_counts(normalize=True)

predicted_distribution = pd.Series(
    v4a_test_predictions
).value_counts(normalize=True)

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED RESULT DISTRIBUTION")
print("=" * 60)

print("\nACTUAL:")
print(actual_distribution.sort_index())

print("\nPREDICTED:")
print(predicted_distribution.sort_index())


# V4-A DRAW PROBABILITY ANALYSIS
v4a_draw_analysis = pd.DataFrame({
    "actual_result": y_v4a_test.values,
    "draw_probability": v4a_draw_prob
})

v4a_draw_analysis["is_actual_draw"] = (
    v4a_draw_analysis["actual_result"] == "D"
)

print("\n" + "=" * 60)
print("V4-A DRAW PROBABILITY ANALYSIS")
print("=" * 60)

print(
    f"Mean Draw probability: "
    f"{v4a_draw_probability_mean if False else v4a_draw_prob.mean():.4f}"
)

print(
    f"Median Draw probability: "
    f"{np.median(v4a_draw_prob):.4f}"
)

print(
    f"Minimum Draw probability: "
    f"{v4a_draw_prob.min():.4f}"
)

print(
    f"Maximum Draw probability: "
    f"{v4a_draw_prob.max():.4f}"
)

print("\nActual draws:", int(v4a_draw_analysis["is_actual_draw"].sum()))
print(
    "Actual draw rate:",
    f"{v4a_draw_analysis['is_actual_draw'].mean():.4f}"
)


# CHECK AVAILABLE MODEL VARIABLES
print("V2-related variables:")

for name in sorted(globals()):
    if "v2" in name.lower() or "prob" in name.lower():
        print(name)


# V2 VS V4-A PROBABILITY COMPARISON
# Get the V2 class order
v2_class_order = list(v2_cb_classes)

v2_d_idx = v2_class_order.index("D")

# V2 Draw probabilities
v2_draw_prob = v2_cb_prob[:, v2_d_idx]


# BUILD COMPARISON DATAFRAME
comparison = pd.DataFrame({
    "actual_result": y_test_v2.values,
    "v2_draw_probability": v2_draw_prob,
    "v4a_draw_probability": v4a_draw_prob
})

comparison["draw_probability_change"] = (
    comparison["v4a_draw_probability"]
    - comparison["v2_draw_probability"]
)


# DISPLAY COMPARISON
print("\n" + "=" * 60)
print("V2 VS V4-A DRAW PROBABILITY")
print("=" * 60)

print(
    f"V2 mean Draw probability:   "
    f"{v2_draw_prob.mean():.4f}"
)

print(
    f"V4-A mean Draw probability: "
    f"{v4a_draw_prob.mean():.4f}"
)

print(
    f"Mean probability change:    "
    f"{comparison['draw_probability_change'].mean():+.4f}"
)

print(
    f"Actual Draw mean V2:        "
    f"{comparison.loc[comparison.actual_result == 'D', 'v2_draw_probability'].mean():.4f}"
)

print(
    f"Actual Draw mean V4-A:      "
    f"{comparison.loc[comparison.actual_result == 'D', 'v4a_draw_probability'].mean():.4f}"
)


print("\n # V2 VS V4-A SEASON-LEVEL ROBUSTNESS ")
# PREPARE V2 AND V4-A PREDICTIONS
# V2 hard predictions
v2_predictions = v2_cb_pred

# V2 probabilities
v2_probabilities = v2_cb_prob

# V4-A hard predictions
v4a_predictions = v4a_test_predictions

# V4-A probabilities
v4a_probabilities = v4a_test_probabilities


# GET SEASON INFORMATION
# The V2 test dataframe contains the same 431-match holdout
season_test = test_df_v2[[
    "season",
    "date",
    "home_team",
    "away_team",
    "result"
]].copy()

season_test = season_test.reset_index(drop=True)


# HELPER FUNCTION
def calculate_multiclass_brier(y_true, probabilities, classes):
    class_to_index = {
        cls: i for i, cls in enumerate(classes)
    }

    y_encoded = np.array([
        [
            1 if class_to_index[actual] == i else 0
            for i in range(len(classes))
        ]
        for actual in y_true
    ])

    return np.mean(
        np.sum(
            (y_encoded - probabilities) ** 2,
            axis=1
        )
    )


# V2 CLASS ORDER
v2_classes = list(v2_cb_classes)

# V4-A class order
v4a_classes = list(v4a_calibrated_model.classes_)


# CALCULATE RESULTS BY SEASON
season_results = []

for season in sorted(season_test["season"].unique()):

    mask = season_test["season"] == season

    y_true = season_test.loc[mask, "result"].values

    # V2
    v2_probs = v2_probabilities[mask.values]
    v2_preds = np.array(v2_predictions)[mask.values]

    v2_accuracy = accuracy_score(
        y_true,
        v2_preds
    )

    v2_logloss = log_loss(
        y_true,
        v2_probs,
        labels=v2_classes
    )

    v2_brier = calculate_multiclass_brier(
        y_true,
        v2_probs,
        v2_classes
    )

    # V4-A
    v4a_probs = v4a_probabilities[mask.values]
    v4a_preds = np.array(v4a_predictions)[mask.values]

    v4a_accuracy = accuracy_score(
        y_true,
        v4a_preds
    )

    v4a_logloss = log_loss(
        y_true,
        v4a_probs,
        labels=v4a_classes
    )

    v4a_brier = calculate_multiclass_brier(
        y_true,
        v4a_probs,
        v4a_classes
    )

    # Store
    season_results.append({
        "season": season,
        "matches": int(mask.sum()),

        "v2_accuracy": v2_accuracy,
        "v4a_accuracy": v4a_accuracy,
        "accuracy_change": v4a_accuracy - v2_accuracy,

        "v2_logloss": v2_logloss,
        "v4a_logloss": v4a_logloss,
        "logloss_change": v4a_logloss - v2_logloss,

        "v2_brier": v2_brier,
        "v4a_brier": v4a_brier,
        "brier_change": v4a_brier - v2_brier
    })


# DISPLAY RESULTS
season_results_df = pd.DataFrame(season_results)

print("\n" + "=" * 80)
print("V2 VS V4-A SEASON-LEVEL ROBUSTNESS")
print("=" * 80)

print(
    season_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)




# V2 VS V4-A PREDICTION ALIGNMENT CHECK
print("\n" + "=" * 60)
print("PREDICTION ALIGNMENT CHECK")
print("=" * 60)

print("V2 test rows:", len(test_df_v2))
print("V4-A test rows:", len(v4a_test_df))

print("\nFirst 5 V2 matches:")
print(
    test_df_v2[
        ["season", "date", "home_team", "away_team", "result"]
    ].head().to_string(index=False)
)

print("\nFirst 5 V4-A matches:")
print(
    v4a_test_df[
        ["season", "date", "home_team", "away_team", "result"]
    ].head().to_string(index=False)
)



# EXACT MATCH ALIGNMENT
v2_keys = test_df_v2[
    ["season", "date", "home_team", "away_team"]
].reset_index(drop=True)

v4a_keys = v4a_test_df[
    ["season", "date", "home_team", "away_team"]
].reset_index(drop=True)

alignment_check = v2_keys.equals(v4a_keys)

print("\nSame match ordering:", alignment_check)

if not alignment_check:
    print("\nFirst mismatches:")

    mismatch_mask = (
        v2_keys.astype(str).ne(v4a_keys.astype(str)).any(axis=1)
    )

    print(
        pd.concat(
            [
                v2_keys[mismatch_mask].add_prefix("V2_"),
                v4a_keys[mismatch_mask].add_prefix("V4A_")
            ],
            axis=1
        ).head(10).to_string(index=False)
    )
else:
    print("✓ V2 and V4-A test matches are perfectly aligned")



print("\n # ALIGN V2 AND V4-A PREDICTIONS BY MATCH")
# ALIGN V2 AND V4-A PREDICTIONS BY MATCH
# V2 prediction dataframe
v2_prediction_df = test_df_v2[
    [
        "season",
        "date",
        "home_team",
        "away_team",
        "result"
    ]
].copy()

v2_prediction_df["v2_prediction"] = np.array(v2_cb_pred)
v2_prediction_df["v2_prob_A"] = v2_cb_prob[:, v2_cb_classes.tolist().index("A")]
v2_prediction_df["v2_prob_D"] = v2_cb_prob[:, v2_cb_classes.tolist().index("D")]
v2_prediction_df["v2_prob_H"] = v2_cb_prob[:, v2_cb_classes.tolist().index("H")]


# V4-A prediction dataframe
v4a_prediction_df = v4a_test_df[
    [
        "season",
        "date",
        "home_team",
        "away_team",
        "result"
    ]
].copy()

v4a_prediction_df["v4a_prediction"] = np.array(v4a_test_predictions)
v4a_prediction_df["v4a_prob_A"] = v4a_test_probabilities[:, a_idx]
v4a_prediction_df["v4a_prob_D"] = v4a_test_probabilities[:, d_idx]
v4a_prediction_df["v4a_prob_H"] = v4a_test_probabilities[:, h_idx]


print("✓ V2 prediction dataframe created")
print("✓ V4-A prediction dataframe created")



# MERGE V2 AND V4-A BY MATCH KEY
match_keys = [
    "season",
    "date",
    "home_team",
    "away_team"
]

aligned_predictions = v2_prediction_df.merge(
    v4a_prediction_df[
        match_keys +
        [
            "v4a_prediction",
            "v4a_prob_A",
            "v4a_prob_D",
            "v4a_prob_H"
        ]
    ],
    on=match_keys,
    how="inner",
    validate="one_to_one"
)

print("\n" + "=" * 60)
print("ALIGNED PREDICTIONS")
print("=" * 60)

print("V2 rows:", len(v2_prediction_df))
print("V4-A rows:", len(v4a_prediction_df))
print("Aligned rows:", len(aligned_predictions))

print(
    "\nMissing V2/V4-A matches:",
    431 - len(aligned_predictions)
)



# VERIFY ALIGNED RESULTS

print("\n" + "=" * 60)
print("ALIGNMENT VERIFICATION")
print("=" * 60)

print("Aligned rows:", len(aligned_predictions))

print(
    "Unique matches:",
    aligned_predictions[
        ["season", "date", "home_team", "away_team"]
    ].drop_duplicates().shape[0]
)

print("✓ V2 and V4-A predictions aligned by exact fixture")
print("✓ All 431 matches matched")



print("\n # V2 VS V4-A — CORRECT SEASON-LEVEL ROBUSTNESS")

# V2 VS V4-A — CORRECT SEASON-LEVEL ROBUSTNESS
def multiclass_brier(y_true, probabilities, classes):
    class_to_index = {cls: i for i, cls in enumerate(classes)}

    y_one_hot = np.zeros((len(y_true), len(classes)))

    for row_index, actual in enumerate(y_true):
        y_one_hot[row_index, class_to_index[actual]] = 1

    return np.mean(np.sum((probabilities - y_one_hot) ** 2, axis=1))


season_results = []

for season, group in aligned_predictions.groupby("season", sort=True):

    y_true = group["result"]


    # V2
    v2_predictions = group["v2_prediction"]

    v2_probabilities = group[
        ["v2_prob_A", "v2_prob_D", "v2_prob_H"]
    ].values

    v2_accuracy = accuracy_score(
        y_true,
        v2_predictions
    )

    v2_log_loss = log_loss(
        y_true,
        v2_probabilities,
        labels=["A", "D", "H"]
    )

    v2_brier = multiclass_brier(
        y_true,
        v2_probabilities,
        ["A", "D", "H"]
    )


    # V4-A
    v4a_predictions = group["v4a_prediction"]

    v4a_probabilities = group[
        ["v4a_prob_A", "v4a_prob_D", "v4a_prob_H"]
    ].values

    v4a_accuracy = accuracy_score(
        y_true,
        v4a_predictions
    )

    v4a_log_loss = log_loss(
        y_true,
        v4a_probabilities,
        labels=["A", "D", "H"]
    )

    v4a_brier = multiclass_brier(
        y_true,
        v4a_probabilities,
        ["A", "D", "H"]
    )

    season_results.append({
        "season": season,
        "matches": len(group),

        "v2_accuracy": v2_accuracy,
        "v4a_accuracy": v4a_accuracy,
        "accuracy_change": v4a_accuracy - v2_accuracy,

        "v2_log_loss": v2_log_loss,
        "v4a_log_loss": v4a_log_loss,
        "log_loss_change": v4a_log_loss - v2_log_loss,

        "v2_brier": v2_brier,
        "v4a_brier": v4a_brier,
        "brier_change": v4a_brier - v2_brier,
    })


season_robustness = pd.DataFrame(season_results)


print("\n" + "=" * 70)
print("V2 VS V4-A — SEASON ROBUSTNESS")
print("=" * 70)

print(
    season_robustness.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("AVERAGE CHANGE")
print("=" * 70)

print(
    "Mean accuracy change:",
    f"{season_robustness['accuracy_change'].mean():.4f}"
)

print(
    "Mean Log Loss change:",
    f"{season_robustness['log_loss_change'].mean():.4f}"
)

print(
    "Mean Brier change:",
    f"{season_robustness['brier_change'].mean():.4f}"
)



# V4-A — DRAW PROBABILITY COMPARISON
draw_matches = aligned_predictions[
    aligned_predictions["result"] == "D"
].copy()

print("\n" + "=" * 70)
print("V2 VS V4-A — ACTUAL DRAW MATCHES")
print("=" * 70)

print("Actual draw matches:", len(draw_matches))

print("\nV2 Draw Probability:")
print(
    draw_matches["v2_prob_D"].describe()
)

print("\nV4-A Draw Probability:")
print(
    draw_matches["v4a_prob_D"].describe()
)

print("\nAverage Draw Probability:")
print(
    "V2:",
    f"{draw_matches['v2_prob_D'].mean():.4f}"
)

print(
    "V4-A:",
    f"{draw_matches['v4a_prob_D'].mean():.4f}"
)

print(
    "Change:",
    f"{draw_matches['v4a_prob_D'].mean() - draw_matches['v2_prob_D'].mean():+.4f}"
)

print("\n" + "=" * 70)
print("DRAW PROBABILITY BY SEASON")
print("=" * 70)

draw_by_season = (
    draw_matches
    .groupby("season")
    .agg(
        matches=("result", "size"),
        v2_mean_draw_probability=("v2_prob_D", "mean"),
        v4a_mean_draw_probability=("v4a_prob_D", "mean"),
    )
    .reset_index()
)

draw_by_season["change"] = (
    draw_by_season["v4a_mean_draw_probability"]
    - draw_by_season["v2_mean_draw_probability"]
)

print(
    draw_by_season.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)



# V4-A.1 — RELATIVE CURRENT-SEASON FEATURES
V4A1_FEATURES = [
    "current_season_win_rate_edge",
    "current_season_draw_rate_edge",
    "current_season_goal_difference_edge",
    "current_season_attack_edge",
    "current_season_defense_edge",
]

V4A1_MODEL_FEATURES = V2_MODEL_FEATURES + V4A1_FEATURES

print("\n" + "=" * 70)
print("V4-A.1 FEATURE SET")
print("=" * 70)

print("V2 features:", len(V2_MODEL_FEATURES))
print("V4-A.1 features:", len(V4A1_FEATURES))
print("Total features:", len(V4A1_MODEL_FEATURES))

print("\nV4-A.1 added features:")
for feature in V4A1_FEATURES:
    print("✓", feature)



# TRAIN V4-A.1
X_v4a1 = features_v4a_combined[V4A1_MODEL_FEATURES].copy()
y_v4a1 = features_v4a_combined["result"].copy()

# SAME TIME-AWARE SPLIT
train_mask_v4a1 = features_v4a_combined["date"] < "2023-11-29"
test_mask_v4a1 = features_v4a_combined["date"] >= "2023-11-29"

X_train_v4a1 = X_v4a1.loc[train_mask_v4a1]
X_test_v4a1 = X_v4a1.loc[test_mask_v4a1]

y_train_v4a1 = y_v4a1.loc[train_mask_v4a1]
y_test_v4a1 = y_v4a1.loc[test_mask_v4a1]

print("\n" + "=" * 70)
print("V4-A.1 TIME-AWARE SPLIT")
print("=" * 70)

print("Train rows:", len(X_train_v4a1))
print("Test rows:", len(X_test_v4a1))

print(
    "Train dates:",
    features_v4a_combined.loc[train_mask_v4a1, "date"].min(),
    "to",
    features_v4a_combined.loc[train_mask_v4a1, "date"].max()
)

print(
    "Test dates:",
    features_v4a_combined.loc[test_mask_v4a1, "date"].min(),
    "to",
    features_v4a_combined.loc[test_mask_v4a1, "date"].max()
)



# CALIBRATED CATBOOST — V4-A.1
catboost_v4a1 = CatBoostClassifier(
#    catboost_model_v2 = CatBoostClassifier(
#    iterations=500,
#    depth=5,
#    learning_rate=0.03,
#    loss_function="MultiClass",
#    random_seed=42,
#    verbose=False
#)
)

time_split_v4a1 = TimeSeriesSplit(n_splits=5)

calibrated_v4a1 = CalibratedClassifierCV(
    estimator=catboost_v4a1,
    cv=time_split_v4a1,
    method="sigmoid",
    ensemble=True
)

calibrated_v4a1.fit(
    X_train_v4a1,
    y_train_v4a1
)

v4a1_test_probabilities = calibrated_v4a1.predict_proba(
    X_test_v4a1
)

v4a1_classes = calibrated_v4a1.classes_

v4a1_test_predictions = calibrated_v4a1.predict(
    X_test_v4a1
)

print("\nV4-A.1 trained successfully")
print("Probability shape:", v4a1_test_probabilities.shape)
print("Classes:", v4a1_classes)



# V4-A.1 EVALUATION
a_idx = v4a1_classes.tolist().index("A")
d_idx = v4a1_classes.tolist().index("D")
h_idx = v4a1_classes.tolist().index("H")

v4a1_accuracy = accuracy_score(
    y_test_v4a1,
    v4a1_test_predictions
)

v4a1_log_loss = log_loss(
    y_test_v4a1,
    v4a1_test_probabilities,
    labels=["A", "D", "H"]
)

v4a1_brier = multiclass_brier(
    y_test_v4a1,
    v4a1_test_probabilities,
    ["A", "D", "H"]
)

v4a1_normalized_brier = v4a1_brier / 3

print("\n" + "=" * 70)
print("V4-A.1 RESULTS")
print("=" * 70)

print("Accuracy:", f"{v4a1_accuracy:.4f}")
print("Log Loss:", f"{v4a1_log_loss:.4f}")
print("Brier:", f"{v4a1_brier:.4f}")
print("Normalized Brier:", f"{v4a1_normalized_brier:.4f}")

print("\nAverage probabilities:")

print(
    "Away:",
    f"{v4a1_test_probabilities[:, a_idx].mean():.4f}"
)

print(
    "Draw:",
    f"{v4a1_test_probabilities[:, d_idx].mean():.4f}"
)

print(
    "Home:",
    f"{v4a1_test_probabilities[:, h_idx].mean():.4f}"
)

print("\nHard predictions:")

print(
    pd.Series(v4a1_test_predictions)
    .value_counts()
    .sort_index()
)



# V4-A — CURRENT-SEASON SAMPLE SIZE DIAGNOSTIC
print("\n" + "=" * 70)
print("CURRENT-SEASON SAMPLE SIZE DIAGNOSTIC")
print("=" * 70)

print("\nHome current-season matches:")
print(
    features_v4a_combined["home_current_season_matches"]
    .value_counts()
    .sort_index()
)

print("\nAway current-season matches:")
print(
    features_v4a_combined["away_current_season_matches"]
    .value_counts()
    .sort_index()
)

print("\nTest-set sample sizes:")

test_v4a_sample = features_v4a_combined.loc[
    test_mask_v4a1,
    [
        "home_current_season_matches",
        "away_current_season_matches"
    ]
]

print(
    test_v4a_sample.describe()
)



# V4-A.2 — STABILIZED CURRENT-SEASON FEATURES
# We keep the original V4-A features.
# These new features stabilize current-season rates by blending
# them with historical performance.

def add_stabilized_current_season_features(df, prior_strength):
    result = df.copy()

    # WIN RATE
    result[f"home_stabilized_win_rate_p{prior_strength}"] = (
        (
            result["home_current_season_win_rate"]
            * result["home_current_season_matches"]
        )
        +
        (
            result["home_win_rate"]
            * prior_strength
        )
    ) / (
        result["home_current_season_matches"]
        + prior_strength
    )

    result[f"away_stabilized_win_rate_p{prior_strength}"] = (
        (
            result["away_current_season_win_rate"]
            * result["away_current_season_matches"]
        )
        +
        (
            result["away_win_rate"]
            * prior_strength
        )
    ) / (
        result["away_current_season_matches"]
        + prior_strength
    )

    # DRAW RATE
    result[f"home_stabilized_draw_rate_p{prior_strength}"] = (
        (
            result["home_current_season_draw_rate"]
            * result["home_current_season_matches"]
        )
        +
        (
            result["home_draw_rate"]
            * prior_strength
        )
    ) / (
        result["home_current_season_matches"]
        + prior_strength
    )

    result[f"away_stabilized_draw_rate_p{prior_strength}"] = (
        (
            result["away_current_season_draw_rate"]
            * result["away_current_season_matches"]
        )
        +
        (
            result["away_draw_rate"]
            * prior_strength
        )
    ) / (
        result["away_current_season_matches"]
        + prior_strength
    )

    # RELATIVE STABILIZED EDGES
    result[
        f"stabilized_win_rate_edge_p{prior_strength}"
    ] = (
        result[f"home_stabilized_win_rate_p{prior_strength}"]
        -
        result[f"away_stabilized_win_rate_p{prior_strength}"]
    )

    result[
        f"stabilized_draw_rate_edge_p{prior_strength}"
    ] = (
        result[f"home_stabilized_draw_rate_p{prior_strength}"]
        -
        result[f"away_stabilized_draw_rate_p{prior_strength}"]
    )

    return result


print(
    [
        col for col in features_v4a_combined.columns
        if "draw_rate" in col
    ]
)



print("\n" + "=" * 70)
print("AVAILABLE HISTORICAL RATE FEATURES")
print("=" * 70)

for col in features_v4a_combined.columns:
    if "rate" in col.lower():
        print(col)

    


# V4-A.2 — TRAINING-ONLY PRIORS
print("\n" + "=" * 70)
print("V4-A.2 TRAINING-ONLY PRIORS")
print("=" * 70)

# Use only the original training period.
# This prevents information from the 431-match test set
# from influencing the new features.

training_v4a2 = features_v4a_combined.loc[
    features_v4a_combined["date"] < "2023-11-29"
].copy()

print("Training rows:", len(training_v4a2))

# HISTORICAL DRAW RATE
historical_draw_rate = (
    training_v4a2["result"] == "D"
).mean()

print(
    "Historical training draw rate:",
    f"{historical_draw_rate:.4f}"
)

# HISTORICAL WIN RATE
print(
    "Average home historical win rate:",
    f"{training_v4a2['home_win_rate'].mean():.4f}"
)

print(
    "Average away historical win rate:",
    f"{training_v4a2['away_win_rate'].mean():.4f}"
)

# PRIOR STRENGTHS
print("\nPrior strengths to test:")
print("✓ 3")
print("✓ 5")
print("✓ 8")




# V4-A.2 — BUILD STABILIZED FEATURES
def add_v4a2_features(df, prior_strength, historical_draw_rate):
    result = df.copy()


    # CURRENT-SEASON WIN RATE — STABILIZED
    result[f"home_stabilized_win_rate_p{prior_strength}"] = (
        (
            result["home_current_season_win_rate"]
            * result["home_current_season_matches"]
        )
        +
        (
            result["home_win_rate"]
            * prior_strength
        )
    ) / (
        result["home_current_season_matches"]
        + prior_strength
    )

    result[f"away_stabilized_win_rate_p{prior_strength}"] = (
        (
            result["away_current_season_win_rate"]
            * result["away_current_season_matches"]
        )
        +
        (
            result["away_win_rate"]
            * prior_strength
        )
    ) / (
        result["away_current_season_matches"]
        + prior_strength
    )


    # CURRENT-SEASON DRAW RATE — STABILIZED
    result[f"home_stabilized_draw_rate_p{prior_strength}"] = (
        (
            result["home_current_season_draw_rate"]
            * result["home_current_season_matches"]
        )
        +
        (
            historical_draw_rate
            * prior_strength
        )
    ) / (
        result["home_current_season_matches"]
        + prior_strength
    )

    result[f"away_stabilized_draw_rate_p{prior_strength}"] = (
        (
            result["away_current_season_draw_rate"]
            * result["away_current_season_matches"]
        )
        +
        (
            historical_draw_rate
            * prior_strength
        )
    ) / (
        result["away_current_season_matches"]
        + prior_strength
    )

    # STABILIZED EDGES
    result[f"stabilized_win_rate_edge_p{prior_strength}"] = (
        result[f"home_stabilized_win_rate_p{prior_strength}"]
        -
        result[f"away_stabilized_win_rate_p{prior_strength}"]
    )

    result[f"stabilized_draw_rate_edge_p{prior_strength}"] = (
        result[f"home_stabilized_draw_rate_p{prior_strength}"]
        -
        result[f"away_stabilized_draw_rate_p{prior_strength}"]
    )

    return result



# CREATE V4-A.2 VARIANTS
v4a2_datasets = {}

for prior_strength in [3, 5, 8]:

    v4a2_datasets[prior_strength] = add_v4a2_features(
        features_v4a_combined,
        prior_strength,
        historical_draw_rate
    )

    print(
        f"✓ V4-A.2 prior strength {prior_strength} created"
    )



# V4-A.2 — FEATURE TRANSFORMATION CHECK
for prior_strength in [3, 5, 8]:

    df = v4a2_datasets[prior_strength]

    print("\n" + "=" * 70)
    print(
        f"PRIOR STRENGTH = {prior_strength}"
    )
    print("=" * 70)

    print(
        "Home stabilized win rate:",
        f"{df[f'home_stabilized_win_rate_p{prior_strength}'].mean():.4f}"
    )

    print(
        "Away stabilized win rate:",
        f"{df[f'away_stabilized_win_rate_p{prior_strength}'].mean():.4f}"
    )

    print(
        "Home stabilized draw rate:",
        f"{df[f'home_stabilized_draw_rate_p{prior_strength}'].mean():.4f}"
    )

    print(
        "Away stabilized draw rate:",
        f"{df[f'away_stabilized_draw_rate_p{prior_strength}'].mean():.4f}"
    )




# V4-A.2 — TRAIN AND EVALUATE ALL PRIOR STRENGTHS
v4a2_results = []
v4a2_predictions = {}

# BASE V4-A FEATURES
V4A_BASE_FEATURES = [
    "home_current_season_matches",
    "away_current_season_matches",
    "home_current_season_win_rate",
    "away_current_season_win_rate",
    "home_current_season_draw_rate",
    "away_current_season_draw_rate",
    "home_current_season_goal_difference",
    "away_current_season_goal_difference",
    "home_current_season_avg_goals_for",
    "away_current_season_avg_goals_for",
    "home_current_season_avg_goals_against",
    "away_current_season_avg_goals_against",
    "current_season_win_rate_edge",
    "current_season_draw_rate_edge",
    "current_season_goal_difference_edge",
    "current_season_attack_edge",
    "current_season_defense_edge",
]

# TEST EACH PRIOR STRENGTH
for prior_strength in [3, 5, 8]:

    print("\n" + "=" * 70)
    print(
        f"TRAINING V4-A.2 — PRIOR STRENGTH {prior_strength}"
    )
    print("=" * 70)

    df = v4a2_datasets[prior_strength]

    stabilized_features = [
        f"home_stabilized_win_rate_p{prior_strength}",
        f"away_stabilized_win_rate_p{prior_strength}",
        f"home_stabilized_draw_rate_p{prior_strength}",
        f"away_stabilized_draw_rate_p{prior_strength}",
        f"stabilized_win_rate_edge_p{prior_strength}",
        f"stabilized_draw_rate_edge_p{prior_strength}",
    ]

    model_features = (
        V2_MODEL_FEATURES
        + V4A_BASE_FEATURES
        + stabilized_features
    )

    X = df[model_features]
    y = df["result"]

    train_mask = df["date"] < "2023-11-29"
    test_mask = df["date"] >= "2023-11-29"

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print("Features:", len(model_features))
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))



    # CATBOOST
    catboost_model_v2 = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.03,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False
)

    calibrated_model =  CalibratedClassifierCV(
    catboost_model_v2,
    cv=TimeSeriesSplit(n_splits=5),
    method="sigmoid",
    ensemble=True
)

    calibrated_model.fit(
        X_train,
        y_train
    )

    probabilities = calibrated_model.predict_proba(
        X_test
    )

    predictions = calibrated_model.predict(
        X_test
    )

    classes = calibrated_model.classes_



    # METRICS
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    logloss = log_loss(
        y_test,
        probabilities,
        labels=["A", "D", "H"]
    )

    brier = multiclass_brier(
        y_test,
        probabilities,
        ["A", "D", "H"]
    )



    # STORE
    v4a2_results.append({
        "prior_strength": prior_strength,
        "features": len(model_features),
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier": brier,
        "normalized_brier": brier / 3,
        "predicted_draws": int(
            np.sum(predictions == "D")
        ),
        "mean_draw_probability": probabilities[
            :, classes.tolist().index("D")
        ].mean(),
    })

    v4a2_predictions[prior_strength] = {
        "model": calibrated_model,
        "predictions": predictions,
        "probabilities": probabilities,
        "classes": classes,
        "y_test": y_test.copy(),
        "test_df": df.loc[test_mask].copy(),
        "features": model_features,
    }

    print(
        "Accuracy:",
        f"{accuracy:.4f}"
    )

    print(
        "Log Loss:",
        f"{logloss:.4f}"
    )

    print(
        "Brier:",
        f"{brier:.4f}"
    )

    print(
        "Normalized Brier:",
        f"{brier / 3:.4f}"
    )

    print(
        "Predicted Draws:",
        int(np.sum(predictions == "D"))
    )

    print(
        "Mean Draw Probability:",
        f"{probabilities[:, classes.tolist().index('D')].mean():.4f}"
    )



# V4-A.2 — COMPARISON
v4a2_results_df = pd.DataFrame(v4a2_results)

print("\n" + "=" * 70)
print("V4-A.2 PRIOR COMPARISON")
print("=" * 70)

print(
    v4a2_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)




# V4-A.2 — SEASON-LEVEL ROBUSTNESS COMPARISON
# V2 predictions
v2_prediction_df = test_df_v2[
    ["season", "date", "home_team", "away_team", "result"]
].copy()

v2_prediction_df["v2_prediction"] = np.array(v2_cb_pred)

v2_prediction_df["v2_prob_A"] = v2_cb_prob[
    :, v2_cb_classes.tolist().index("A")
]

v2_prediction_df["v2_prob_D"] = v2_cb_prob[
    :, v2_cb_classes.tolist().index("D")
]

v2_prediction_df["v2_prob_H"] = v2_cb_prob[
    :, v2_cb_classes.tolist().index("H")
]


# V4-A.2 predictions
match_keys = [
    "season",
    "date",
    "home_team",
    "away_team"
]

v4a2_comparison_list = []

for prior_strength, prediction_data in v4a2_predictions.items():

    temp = prediction_data["test_df"][
        match_keys + ["result"]
    ].copy()

    temp["v4a2_prediction"] = np.array(
        prediction_data["predictions"]
    )

    classes = prediction_data["classes"]
    probabilities = prediction_data["probabilities"]

    temp["v4a2_prob_A"] = probabilities[
        :, classes.tolist().index("A")
    ]

    temp["v4a2_prob_D"] = probabilities[
        :, classes.tolist().index("D")
    ]

    temp["v4a2_prob_H"] = probabilities[
        :, classes.tolist().index("H")
    ]

    temp["prior_strength"] = prior_strength

    v4a2_comparison_list.append(temp)


# ALIGN V2 WITH EACH V4-A.2 VERSION
season_results = []

for v4a2_df in v4a2_comparison_list:

    prior_strength = v4a2_df["prior_strength"].iloc[0]

    aligned = v2_prediction_df.merge(
        v4a2_df[
            match_keys
            + [
                "v4a2_prediction",
                "v4a2_prob_A",
                "v4a2_prob_D",
                "v4a2_prob_H"
            ]
        ],
        on=match_keys,
        how="inner",
        validate="one_to_one"
    )

    print(
        f"\nPrior strength {prior_strength}: "
        f"{len(aligned)} aligned matches"
    )

    for season in sorted(aligned["season"].unique()):

        season_df = aligned[
            aligned["season"] == season
        ].copy()

        y_true = season_df["result"]

        v2_pred = season_df["v2_prediction"]

        v4a2_pred = season_df["v4a2_prediction"]

        v2_prob = season_df[
            ["v2_prob_A", "v2_prob_D", "v2_prob_H"]
        ].values

        v4a2_prob = season_df[
            ["v4a2_prob_A", "v4a2_prob_D", "v4a2_prob_H"]
        ].values

        v2_accuracy = accuracy_score(
            y_true,
            v2_pred
        )

        v4a2_accuracy = accuracy_score(
            y_true,
            v4a2_pred
        )

        v2_logloss = log_loss(
            y_true,
            v2_prob,
            labels=["A", "D", "H"]
        )

        v4a2_logloss = log_loss(
            y_true,
            v4a2_prob,
            labels=["A", "D", "H"]
        )

        v2_brier = multiclass_brier(
            y_true,
            v2_prob,
            ["A", "D", "H"]
        )

        v4a2_brier = multiclass_brier(
            y_true,
            v4a2_prob,
            ["A", "D", "H"]
        )

        season_results.append({
            "prior_strength": prior_strength,
            "season": season,
            "matches": len(season_df),

            "v2_accuracy": v2_accuracy,
            "v4a2_accuracy": v4a2_accuracy,
            "accuracy_change": (
                v4a2_accuracy - v2_accuracy
            ),

            "v2_log_loss": v2_logloss,
            "v4a2_log_loss": v4a2_logloss,
            "log_loss_change": (
                v4a2_logloss - v2_logloss
            ),

            "v2_brier": v2_brier,
            "v4a2_brier": v4a2_brier,
            "brier_change": (
                v4a2_brier - v2_brier
            ),

            "v4a2_predicted_draws": int(
                np.sum(v4a2_pred == "D")
            )
        })


# DISPLAY RESULTS
v4a2_season_results_df = pd.DataFrame(
    season_results
)

print("\n" + "=" * 90)
print("V4-A.2 — SEASON ROBUSTNESS")
print("=" * 90)

print(
    v4a2_season_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)




# V4-A.2 — SAMPLE SIZE ROBUSTNESS ANALYSIS
# We will analyze the best current version: prior strength 8

prior_strength = 8

v4a2_data = v4a2_predictions[prior_strength]

v4a2_test = v4a2_data["test_df"].copy()

v4a2_test["v4a2_prediction"] = np.array(
    v4a2_data["predictions"]
)

v4a2_probabilities = v4a2_data["probabilities"]
v4a2_classes = v4a2_data["classes"]

v4a2_test["v4a2_prob_A"] = v4a2_probabilities[
    :, v4a2_classes.tolist().index("A")
]

v4a2_test["v4a2_prob_D"] = v4a2_probabilities[
    :, v4a2_classes.tolist().index("D")
]

v4a2_test["v4a2_prob_H"] = v4a2_probabilities[
    :, v4a2_classes.tolist().index("H")
]



# ADD CURRENT-SEASON SAMPLE SIZE
v4a2_test["combined_current_matches"] = (
    v4a2_test["home_current_season_matches"]
    +
    v4a2_test["away_current_season_matches"]
)



# CREATE SAMPLE-SIZE GROUPS
def sample_size_group(value):

    if value <= 2:
        return "0-2"

    elif value <= 5:
        return "3-5"

    elif value <= 8:
        return "6-8"

    else:
        return "9+"


v4a2_test["sample_size_group"] = (
    v4a2_test["combined_current_matches"]
    .apply(sample_size_group)
)



# SAMPLE-SIZE PERFORMANCE
sample_size_results = []

for group in ["0-2", "3-5", "6-8", "9+"]:

    group_df = v4a2_test[
        v4a2_test["sample_size_group"] == group
    ].copy()

    if len(group_df) == 0:
        continue

    y_true = group_df["result"]

    predictions = group_df[
        "v4a2_prediction"
    ]

    probabilities = group_df[
        [
            "v4a2_prob_A",
            "v4a2_prob_D",
            "v4a2_prob_H"
        ]
    ].values

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    logloss = log_loss(
        y_true,
        probabilities,
        labels=["A", "D", "H"]
    )

    brier = multiclass_brier(
        y_true,
        probabilities,
        ["A", "D", "H"]
    )

    sample_size_results.append({
        "sample_size_group": group,
        "matches": len(group_df),
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier": brier,
        "mean_draw_probability": group_df[
            "v4a2_prob_D"
        ].mean(),
        "predicted_draws": int(
            np.sum(predictions == "D")
        )
    })



# DISPLAY
sample_size_results_df = pd.DataFrame(
    sample_size_results
)

print("\n" + "=" * 90)
print("V4-A.2 — CURRENT-SEASON SAMPLE SIZE ANALYSIS")
print("=" * 90)

print(
    sample_size_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)