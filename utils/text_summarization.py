

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt_tab')


# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def summarize_text(text, num_sentences=3):
    """
    Generate an extractive summary of the text.
    
    Parameters:
    text (str): Input text to summarize
    num_sentences (int): Number of sentences in the summary
    
    Returns:
    str: Summarized text
    """
    if not isinstance(text, str) or text.strip() == "":
        return "No text available to summarize."
    
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # If there are fewer sentences than requested, return the original text
    if len(sentences) <= num_sentences:
        return text
    
    # Create a TF-IDF vectorizer to identify important sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on the sentences
    try:
        X = vectorizer.fit_transform(sentences)
        
        # Calculate sentence importance based on the sum of TF-IDF scores of its words
        importance = np.array([sum(X[i].toarray()[0]) for i in range(len(sentences))])
        
        # Get the indices of the most important sentences
        top_indices = importance.argsort()[-num_sentences:]
        top_indices = sorted(top_indices)  # Sort to maintain original order
        
        # Construct the summary from the top sentences
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    except:
        # Fallback if vectorization fails (e.g., very short text)
        return ' '.join(sentences[:num_sentences])

def summarize_top_posts(df, num_posts=5, num_sentences=3):
    """
    Summarize the top posts in the dataset.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    num_posts (int): Number of top posts to summarize
    num_sentences (int): Number of sentences per summary
    
    Returns:
    pandas.DataFrame: DataFrame with post titles and summaries
    """
    # Sort posts by upvotes (descending)
    top_posts = df.sort_values('ups', ascending=False).head(num_posts)
    
    # Combine title and selftext for summarization
    top_posts['full_text'] = top_posts['title'] + '. ' + top_posts['selftext'].fillna('')
    
    # Generate summaries
    top_posts['summary'] = top_posts['full_text'].apply(lambda x: summarize_text(x, num_sentences))
    
    # Return relevant columns
    return top_posts[['title', 'summary', 'ups', 'num_comments', 'author', 'subreddit']]