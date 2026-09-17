# IMPORT LIBRARIES
import pandas as pd
import re
from pathlib import Path


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