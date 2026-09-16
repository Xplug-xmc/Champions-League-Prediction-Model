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