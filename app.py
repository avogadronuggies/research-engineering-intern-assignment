import streamlit as st
import os
from utils.data_loader import load_dataset
from utils.sentiment_analysis import analyze_sentiment, plot_sentiment_trend
from utils.text_preprocessing import preprocess_text
from utils.visualization import (
    generate_wordcloud, 
    plot_community_pie_chart,
    plot_upvotes_scatter,
    plot_word_frequency_pie
)
from utils.topic_modeling import perform_topic_modeling
from utils.network_analysis import plot_network_graph
from utils.text_summarization import summarize_top_posts
from utils.user_activity_analysis import analyze_user_activity
from utils.content_filters import add_filter_controls
from utils.export_functionality import export_data_and_charts
from utils.emoji_analysis import analyze_emojis
from collections import Counter
import pandas as pd
import plotly.express as px
import json

# Custom CSS for styling (unchanged)
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

# Function to load data from data.jsonl file
def load_data_jsonl(file_path="data\data.jsonl"):
    """
    Load data from a specific JSONL file.
    
    Parameters:
    file_path (str): Path to the JSONL file
    
    Returns:
    pandas.DataFrame: Loaded dataset or None if file not found
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = [json.loads(line) for line in file]
                df = pd.DataFrame([post['data'] for post in data])
                return df
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Generate custom insights function (unchanged)
def generate_custom_insights(df):
    """Generate custom insights based on the dataset."""
    # Analyze sentiment distribution
    sentiment_distribution = df['sentiment'].value_counts(normalize=True) * 100
    sentiment_insight = f"Sentiment Analysis:\n- Positive: {sentiment_distribution.get('positive', 0):.2f}%\n- Neutral: {sentiment_distribution.get('neutral', 0):.2f}%\n- Negative: {sentiment_distribution.get('negative', 0):.2f}%"
    
    # Analyze top communities
    top_communities = df['subreddit'].value_counts().head(3).index.tolist()
    communities_insight = f"Top Communities:\n- {', '.join(top_communities)}"
    
    # Analyze word cloud
    all_titles = ' '.join(df['title'].fillna(''))
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

    # Info message about data source
    st.info("This application automatically loads data from 'data.jsonl'")
    
    # Load the dataset from data.jsonl
    df = load_data_jsonl()
    
    if df is not None:
        # Perform sentiment analysis
        df = analyze_sentiment(df)
        
        # Ensure created_date exists
        if 'created_date' not in df.columns:
            df['created_date'] = pd.to_datetime(df['created'], unit='s')
        
        # Apply filters
        filtered_df = add_filter_controls(df)
        
        # Display dataset info
        st.subheader("Dataset Overview")
        st.write(f"Analyzing {len(filtered_df)} posts from {filtered_df['subreddit'].nunique()} subreddits")
        
        # Create tabs for different analyses
        tabs = st.tabs([
            "Sentiment Analysis", 
            "Text Analysis", 
            "Community Analysis", 
            "User Analysis", 
            "Topic Modeling",
            "Export Data"
        ])
        
        # Tab 1: Sentiment Analysis
        with tabs[0]:
            st.header("Sentiment Analysis")
            
            # Display sentiment distribution
            sentiment_distribution = filtered_df['sentiment'].value_counts()
            st.subheader("Sentiment Distribution")
            st.write(sentiment_distribution)
            
            # Plot sentiment trend over time
            st.subheader("Sentiment Trend Over Time")
            plot_sentiment_trend(filtered_df)
            
            # Display top positive and negative posts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Positive Posts")
                st.write(filtered_df[filtered_df['sentiment'] == 'positive'].sort_values('sentiment_score', ascending=False)[['title', 'sentiment_score']].head())
            
            with col2:
                st.subheader("Top Negative Posts")
                st.write(filtered_df[filtered_df['sentiment'] == 'negative'].sort_values('sentiment_score')[['title', 'sentiment_score']].head())
            
            # Scatterplot: Upvotes vs Sentiment Score
            st.subheader("Upvotes vs Sentiment Score")
            plot_upvotes_scatter(filtered_df, y_axis='sentiment_score')
        
        # Tab 2: Text Analysis
        with tabs[1]:
            st.header("Text Analysis")
            
            # Generate word cloud
            st.subheader("Word Cloud of Post Titles")
            wordcloud_img = generate_wordcloud(filtered_df)
            st.image(wordcloud_img, use_container_width=True)
            
            # Display top 10 most frequent words as a pie chart
            st.subheader("Top 10 Most Frequent Words")
            all_titles = ' '.join(filtered_df['title'].fillna(''))
            filtered_words = preprocess_text(all_titles)
            word_counts = Counter(filtered_words)
            top_words = word_counts.most_common(10)
            plot_word_frequency_pie(top_words)
            
            # Text summarization
            st.subheader("Summaries of Top Posts")
            num_posts = st.slider("Number of posts to summarize", min_value=1, max_value=10, value=5)
            num_sentences = st.slider("Sentences per summary", min_value=1, max_value=5, value=3)
            
            summaries = summarize_top_posts(filtered_df, num_posts=num_posts, num_sentences=num_sentences)
            for idx, row in summaries.iterrows():
                with st.expander(f"{row['title']} (u/{row['author']})"):
                    st.write(f"**Summary:** {row['summary']}")
                    st.write(f"**Upvotes:** {row['ups']} | **Comments:** {row['num_comments']} | **Subreddit:** r/{row['subreddit']}")
            
            # Emoji analysis
            st.subheader("Emoji Analysis")
            analyze_emojis(filtered_df)
        
        # Tab 3: Community Analysis
        with tabs[2]:
            st.header("Community Analysis")
            
            # Pie chart of communities/accounts
            st.subheader("Top Communities/Accounts")
            plot_community_pie_chart(filtered_df)
            
            # Network visualization
            st.subheader("Network Visualization")
            keyword = st.text_input("Enter a keyword, hashtag, or URL for network visualization")
            if keyword:
                plot_network_graph(filtered_df, keyword)
        
        # Tab 4: User Analysis
        with tabs[3]:
            st.header("User Activity Analysis")
            analyze_user_activity(filtered_df)
        
        # Tab 5: Topic Modeling
        with tabs[4]:
            st.header("Topic Modeling")
            
            # Perform topic modeling
            num_topics = st.slider("Number of topics", min_value=3, max_value=10, value=5)
            topics, filtered_df = perform_topic_modeling(filtered_df, num_topics=num_topics)
            
            # Display key topics
            st.subheader("Key Topics")
            for topic in topics:
                st.write(topic)
            
            # Display time series of key topics
            st.subheader("Time Series of Key Topics")
            topic_time_series = filtered_df.groupby([pd.Grouper(key='created_date', freq='M'), 'dominant_topic']).size().unstack(fill_value=0)
            
            # Convert DataFrame to Plotly figure
            fig = px.line(
                topic_time_series, 
                x=topic_time_series.index, 
                y=topic_time_series.columns,
                title='Time Series of Key Topics',
                labels={'value': 'Number of Posts', 'variable': 'Topic', 'created_date': 'Date'}
            )
            st.plotly_chart(fig)
            
            # Search functionality
            st.subheader("Search Posts")
            query = st.text_input("Enter a keyword or phrase")
            if query:
                filtered_df['contains_query'] = filtered_df['title'].apply(lambda x: query.lower() in x.lower() if isinstance(x, str) else False)
                time_series_data = filtered_df[filtered_df['contains_query']].groupby(pd.Grouper(key='created_date', freq='M')).size().reset_index(name='count')
                
                if not time_series_data.empty:
                    fig = px.line(
                        time_series_data, 
                        x='created_date', 
                        y='count',
                        title=f'Time Series of Posts Containing "{query}"',
                        labels={'created_date': 'Date', 'count': 'Number of Posts'}
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning(f"No posts found containing '{query}'")
        
        # Tab 6: Export Data
        with tabs[5]:
            st.header("Export Data")
            export_data_and_charts(filtered_df)
        
        # Generate custom insights
        st.subheader("Custom Insights")
        insights = generate_custom_insights(filtered_df)
        st.write(insights)
    else:
        st.error("Failed to load data from 'data.jsonl'. Please ensure the file exists in the application directory.")

# Run the app
if __name__ == "__main__":
    main()