from fastapi import FastAPI
import pandas as pd

app = FastAPI()
df = pd.read_csv("../data/cleaned_data.csv")

@app.get("/")
def read_root():
    return {"message": "Reddit Analysis API"}

@app.get("/trending_keywords")
def trending_keywords():
    words = " ".join(df["body"].dropna()).split()
    word_freq = pd.Series(words).value_counts().head(10).to_dict()
    return {"trending_keywords": word_freq}
