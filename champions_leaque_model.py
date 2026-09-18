# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import re
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