import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/labeled_data.csv")

st.title("Reddit Social Media Dashboard")

# Time Series Plot
st.subheader("Post Activity Over Time")
df["created_utc"] = pd.to_datetime(df["created_utc"])
df.set_index("created_utc")["score"].resample("D").count().plot(figsize=(10, 4))
st.pyplot(plt)

# Trending Keywords
st.subheader("Trending Keywords")
st.write(df["body"].str.split().explode().value_counts().head(10))

# Search Function
query = st.text_input("Search Keyword:")
if query:
    results = df[df["body"].str.contains(query, case=False, na=False)]
    st.write(results[["created_utc", "author", "body"]].head(10))
