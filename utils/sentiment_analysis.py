import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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