import streamlit as st
import pandas as pd
import json
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
import io
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
from pyvis.network import Network

# Download the VADER lexicon and stopwords
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextInput input {
        border-radius: 5px;
        padding: 10px;
    }
    .stFileUploader label {
        font-size: 16px;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    .stMarkdown h2 {
        color: #2E86C1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the dataset
def load_dataset(file):
    """Load the dataset from a file (JSONL or CSV)."""
    if file.name.endswith('.jsonl'):
        data = [json.loads(line) for line in file]
        df = pd.DataFrame([post['data'] for post in data])
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        st.error("Unsupported file format. Please upload a JSONL or CSV file.")
        return None
    return df

# Preprocess text
def preprocess_text(text):
    """Preprocess text by removing stopwords and non-alphabetic characters."""
    # Load stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove non-alphabetic characters and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stopwords
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]  # Exclude short words
    return filtered_words

# Perform sentiment analysis
def analyze_sentiment(df):
    """Perform sentiment analysis on post titles using VADER."""
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

# Generate word cloud
def generate_wordcloud(df):
    """Generate a word cloud from post titles."""
    all_titles = ' '.join(df['title'])
    filtered_words = preprocess_text(all_titles)
    filtered_text = ' '.join(filtered_words)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(filtered_text)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return img

# Perform topic modeling
def perform_topic_modeling(df, num_topics=5):
    """Perform topic modeling using NMF."""
    # Preprocess titles
    df['tokens'] = df['title'].apply(lambda x: ' '.join(preprocess_text(x)))
    
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

# Pie chart of communities/accounts
def plot_community_pie_chart(df):
    """Plot a pie chart of top communities/accounts."""
    top_communities = df['subreddit'].value_counts().head(10)
    fig = px.pie(names=top_communities.index, values=top_communities.values, title="Top Communities/Accounts")
    st.plotly_chart(fig)

# Network visualization
def plot_network_graph(df, keyword):
    """Plot a network graph of accounts sharing a specific keyword."""
    # Filter posts containing the keyword
    df_filtered = df[df['title'].str.contains(keyword, case=False, na=False)]
    
    # Create a network graph
    G = nx.Graph()
    for _, row in df_filtered.iterrows():
        G.add_node(row['author'], label=row['author'])
        G.add_edge(row['author'], row['subreddit'])
    
    # Visualize the graph using PyVis
    net = Network(notebook=True, height="500px", width="100%")
    net.from_nx(G)
    net.show("network.html")
    st.components.v1.html(open("network.html", "r").read(), height=500)

# Scatterplot to show upvotes vs another variable
def plot_upvotes_scatter(df, y_axis):
    """Plot a scatterplot of upvotes vs another variable."""
    fig = px.scatter(df, x='ups', y=y_axis, title=f"Upvotes vs {y_axis.capitalize()}",
                     labels={'ups': 'Upvotes', y_axis: y_axis.capitalize()})
    st.plotly_chart(fig)

# Generate custom insights for Reddit data
def generate_custom_insights(df):
    """Generate custom insights based on the dataset."""
    # Analyze sentiment distribution
    sentiment_distribution = df['sentiment'].value_counts(normalize=True) * 100
    sentiment_insight = f"Sentiment Analysis:\n- Positive: {sentiment_distribution.get('positive', 0):.2f}%\n- Neutral: {sentiment_distribution.get('neutral', 0):.2f}%\n- Negative: {sentiment_distribution.get('negative', 0):.2f}%"
    
    # Analyze top communities
    top_communities = df['subreddit'].value_counts().head(3).index.tolist()
    communities_insight = f"Top Communities:\n- {', '.join(top_communities)}"
    
    # Analyze word cloud
    all_titles = ' '.join(df['title'])
    filtered_words = preprocess_text(all_titles)
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(5)
    wordcloud_insight = f"Top Words in Titles:\n- {', '.join([word for word, _ in top_words])}"
    
    # Combine insights
    insights = f"{sentiment_insight}\n\n{communities_insight}\n\n{wordcloud_insight}"
    return insights

# Main function
def main():
    st.title("Reddit Data Analysis Dashboard")

    # Sidebar for user inputs
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a JSONL or CSV file", type=["jsonl", "csv"])
    
    if uploaded_file is not None:
        # Load the dataset
        df = load_dataset(uploaded_file)
        
        if df is not None:
            # Perform sentiment analysis
            df = analyze_sentiment(df)
            
            # Display sentiment distribution
            st.subheader("Sentiment Analysis")
            sentiment_distribution = df['sentiment'].value_counts()
            st.write(sentiment_distribution)
            
            # Display top positive and negative posts
            st.subheader("Top Positive Posts")
            st.write(df[df['sentiment'] == 'positive'][['title', 'sentiment_score']].head())
            
            st.subheader("Top Negative Posts")
            st.write(df[df['sentiment'] == 'negative'][['title', 'sentiment_score']].head())
            
            # Generate word cloud
            st.subheader("Word Cloud of Post Titles")
            wordcloud_img = generate_wordcloud(df)
            st.image(wordcloud_img, use_container_width=True)
            
            # Display top 10 most frequent words as a pie chart
            st.subheader("Top 10 Most Frequent Words")
            all_titles = ' '.join(df['title'])
            filtered_words = preprocess_text(all_titles)
            word_counts = Counter(filtered_words)
            top_words = word_counts.most_common(10)
            
            # Prepare data for the pie chart
            words, counts = zip(*top_words)
            fig = px.pie(names=words, values=counts, title="Top 10 Most Frequent Words")
            st.plotly_chart(fig)
            
            # Pie chart of communities/accounts
            st.subheader("Top Communities/Accounts")
            plot_community_pie_chart(df)
            
            # Scatterplot: Upvotes vs Number of Comments
            st.subheader("Upvotes vs Number of Comments")
            plot_upvotes_scatter(df, y_axis='num_comments')
            
            # Scatterplot: Upvotes vs Sentiment Score
            st.subheader("Upvotes vs Sentiment Score")
            plot_upvotes_scatter(df, y_axis='sentiment_score')
            
            # Network visualization
            st.subheader("Network Visualization")
            keyword = st.text_input("Enter a keyword, hashtag, or URL for network visualization")
            if keyword:
                plot_network_graph(df, keyword)
            
            # Perform topic modeling
            topics, df = perform_topic_modeling(df)
            
            # Display key topics
            st.subheader("Key Topics")
            for topic in topics:
                st.write(topic)
            
            # Display time series of key topics
            st.subheader("Time Series of Key Topics")
            df['created_date'] = pd.to_datetime(df['created'], unit='s')
            topic_time_series = df.groupby([pd.Grouper(key='created_date', freq='M'), 'dominant_topic']).size().unstack(fill_value=0)
            fig = px.line(topic_time_series, x=topic_time_series.index, y=topic_time_series.columns,
                          title='Time Series of Key Topics',
                          labels={'value': 'Number of Posts', 'created_date': 'Date'})
            st.plotly_chart(fig)
            
            # Search functionality
            st.subheader("Search Posts")
            query = st.text_input("Enter a keyword or phrase")
            if query:
                df['contains_query'] = df['title'].apply(lambda x: query.lower() in x.lower())
                time_series_data = df[df['contains_query']].groupby(pd.Grouper(key='created_date', freq='M')).size().reset_index(name='count')
                fig = px.line(time_series_data, x='created_date', y='count',
                              title=f'Time Series of Posts Containing "{query}"',
                              labels={'created_date': 'Date', 'count': 'Number of Posts'})
                st.plotly_chart(fig)
            
            # Generate custom insights
            st.subheader("Custom Insights")
            insights = generate_custom_insights(df)
            st.write(insights)
    else:
        st.info("Please upload a dataset to get started.")

# Run the app
if __name__ == "__main__":
    main()