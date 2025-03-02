from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from utils.text_preprocessing import preprocess_text

def perform_topic_modeling(df, num_topics=5):
    """
    Perform topic modeling using Non-negative Matrix Factorization (NMF).
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    num_topics (int): Number of topics to extract
    
    Returns:
    tuple: (list of topic descriptions, DataFrame with topic assignments)
    """
    # Preprocess titles
    df['tokens'] = df['title'].fillna('').apply(lambda x: ' '.join(preprocess_text(x)))
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(df['tokens'])
    
    # Perform NMF
    nmf = NMF(n_components=num_topics, random_state=42)
    nmf.fit(tfidf)
    
    # Get the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_words))
    
    # Assign dominant topic to each post
    df['dominant_topic'] = nmf.transform(tfidf).argmax(axis=1)
    
    return topics, df


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def perform_topic_modeling(df, num_topics=5):
    """Perform topic modeling using NMF."""
    df['tokens'] = df['title'].apply(lambda x: ' '.join(preprocess_text(x)))
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(df['tokens'])
    nmf = NMF(n_components=num_topics, random_state=42)
    nmf.fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_words))
    df['dominant_topic'] = nmf.transform(tfidf).argmax(axis=1)
    return topics, df