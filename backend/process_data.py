import pandas as pd
import json
from datetime import datetime

def load_and_clean_data(filepath):
    # Read JSONL file
    comments = [json.loads(line) for line in open(filepath, "r", encoding="utf-8")]

    # Normalize nested "data" field
    df = pd.json_normalize([c["data"] for c in comments])

    # Keep relevant columns
    df = df[["author", "created_utc", "selftext", "score", "subreddit"]]

    # Convert timestamp
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")

    # Rename for clarity
    df.rename(columns={"selftext": "body"}, inplace=True)

    # Remove bots & deleted users
    df = df[~df["author"].str.contains("bot|deleted", na=False, case=False)]

    return df

if __name__ == "__main__":
    df_cleaned = load_and_clean_data("../data/data.jsonl")
    df_cleaned.to_csv("../data/cleaned_data.csv", index=False)
    print("Data cleaned and saved.")
