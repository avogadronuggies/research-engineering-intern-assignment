import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text by removing stopwords and non-alphabetic characters.
    
    Parameters:
    text (str): Input text to preprocess
    
    Returns:
    list: List of filtered words
    """
    # Handle non-string inputs
    if not isinstance(text, str):
        return []
        
    # Load stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove non-alphabetic characters and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stopwords
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]  # Exclude short words
    return filtered_words