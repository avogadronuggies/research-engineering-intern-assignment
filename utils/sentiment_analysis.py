import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import pandas as pd
import streamlit as st


# Download the VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_sentiment(df):
    """
    Perform sentiment analysis on post titles using VADER.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    
    Returns:
    pandas.DataFrame: DataFrame with sentiment analysis results
    """
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        if not isinstance(text, str) or text.strip() == "":
            return 0  # Neutral if no text
        return analyzer.polarity_scores(text)['compound']
    
    # Combine title and selftext, handling missing values
    df['text'] = df[['title', 'selftext']].fillna('').agg(' '.join, axis=1)
    df['sentiment_score'] = df['text'].apply(get_sentiment)
    
    # Categorize sentiment
    df['sentiment'] = df['sentiment_score'].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )
    
    return df


def plot_sentiment_trend(df):
    """
    Plot a line chart showing sentiment trends over time.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts with sentiment analysis
    """
    # Ensure we have a datetime column
    if 'created_date' not in df.columns:
        df['created_date'] = pd.to_datetime(df['created'], unit='s')
    
    # Group by date and calculate average sentiment score
    sentiment_by_date = df.groupby(pd.Grouper(key='created_date', freq='D')).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        post_count=('id', 'count')
    ).reset_index()
    
    # Only include dates with sufficient data (at least 3 posts)
    filtered_sentiment = sentiment_by_date[sentiment_by_date['post_count'] >= 3]
    
    # Create the line chart
    fig = px.line(
        filtered_sentiment, 
        x='created_date', 
        y='avg_sentiment',
        title='Sentiment Trend Over Time',
        labels={'created_date': 'Date', 'avg_sentiment': 'Average Sentiment Score'},
        hover_data=['post_count']
    )
    
    # Add a reference line at y=0 (neutral sentiment)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    
    # Color regions based on sentiment
    fig.add_hrect(y0=0.05, y1=1, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Positive")
    fig.add_hrect(y0=-0.05, y1=0.05, line_width=0, fillcolor="gray", opacity=0.1, annotation_text="Neutral")
    fig.add_hrect(y0=-1, y1=-0.05, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Negative")
    
    st.plotly_chart(fig)
    
    

__all__ = ["analyze_sentiment", "plot_sentiment_trend"]
