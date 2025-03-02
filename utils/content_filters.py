import pandas as pd
import streamlit as st



def add_filter_controls(df):
    """
    Add filter controls for the dataset and return the filtered dataframe.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    
    Returns:
    pandas.DataFrame: Filtered DataFrame
    """
    st.sidebar.header("Filter Options")
    
    # Date range filter
    st.sidebar.subheader("Date Range")
    
    # Calculate min and max dates from the data
    if 'created_date' not in df.columns:
        df['created_date'] = pd.to_datetime(df['created'], unit='s')
    
    min_date = df['created_date'].min().date()
    max_date = df['created_date'].max().date()
    
    # Default to the last 30 days if the date range is large
    default_start = max(min_date, (max_date - pd.Timedelta(days=30)))
    
    start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    # Subreddit filter
    st.sidebar.subheader("Subreddit")
    # Get the list of subreddits
    subreddits = df['subreddit'].unique().tolist()
    # Add "All" option at the beginning
    subreddit_options = ["All"] + subreddits
    selected_subreddit = st.sidebar.selectbox("Select Subreddit", subreddit_options)
    
    # Upvotes threshold
    st.sidebar.subheader("Minimum Upvotes")
    min_upvotes = st.sidebar.slider("Minimum Upvotes", min_value=0, max_value=df['ups'].max(), value=0)
    
    # Sentiment filter
    st.sidebar.subheader("Sentiment")
    sentiment_options = ["All", "Positive", "Neutral", "Negative"]
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiment_options)
    
    # Apply filters
    filtered_df = df.copy()
    
    # Date filter
    filtered_df = filtered_df[(filtered_df['created_date'].dt.date >= start_date) & 
                             (filtered_df['created_date'].dt.date <= end_date)]
    
    # Subreddit filter
    if selected_subreddit != "All":
        filtered_df = filtered_df[filtered_df['subreddit'] == selected_subreddit]
    
    # Upvotes filter
    filtered_df = filtered_df[filtered_df['ups'] >= min_upvotes]
    
    # Sentiment filter
    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df['sentiment'].str.lower() == selected_sentiment.lower()]
    
    # Display filter summary
    st.sidebar.subheader("Filter Summary")
    st.sidebar.write(f"Showing {len(filtered_df)} out of {len(df)} posts")
    
    return filtered_df