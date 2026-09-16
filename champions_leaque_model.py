# IMPORT LIBRARIES
import pandas as pd
import re


# LOAD DATASET
df = pd.read_csv("champions_league_2024_25_fbref_raw.csv")


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
df = pd.read_csv("champions_league_2024_25_fbref_raw.csv")


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
df_2023_24 = pd.read_csv("champions_league_2023_24_fbref_raw.csv")


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
def clean_season_data(raw_df):
    # Make a copy so the original raw data is not changed
    df = raw_df.copy()

    # Standardize the team column names
    df = df.rename(
        columns={
            "home": "home_team",
            "away": "away_team"
        }
    )

    # Extract home and away goals from the score column
    df[["home_goals", "away_goals"]] = df["score"].apply(
        lambda x: pd.Series(extract_goals(x))
    )

    # Create the H/D/A result column
    df["result"] = df.apply(
        get_result,
        axis=1
    )

    # Clean home team names
    df["home_team"] = df["home_team"].apply(
        clean_team_name
    )

    # Clean away team names
    df["away_team"] = df["away_team"].apply(
        clean_team_name
    )

    # Keep only the columns needed for our clean dataset
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