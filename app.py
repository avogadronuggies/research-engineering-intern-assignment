import streamlit as st
from utils.data_loader import load_dataset
from utils.sentiment_analysis import analyze_sentiment
from utils.text_preprocessing import preprocess_text
from utils.visualization import (
    generate_wordcloud, 
    plot_community_pie_chart,
    plot_upvotes_scatter,
    plot_word_frequency_pie
)
from utils.topic_modeling import perform_topic_modeling
from utils.network_analysis import plot_network_graph
from collections import Counter
import pandas as pd
import plotly.express as px

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
            all_titles = ' '.join(df['title'].fillna(''))
            filtered_words = preprocess_text(all_titles)
            word_counts = Counter(filtered_words)
            top_words = word_counts.most_common(10)
            
            # Create pie chart for word frequency
            plot_word_frequency_pie(top_words)
            
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
                df['contains_query'] = df['title'].apply(lambda x: query.lower() in x.lower() if isinstance(x, str) else False)
                time_series_data = df[df['contains_query']].groupby(pd.Grouper(key='created_date', freq='M')).size().reset_index(name='count')
                
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
            
            # Generate custom insights
            st.subheader("Custom Insights")
            insights = generate_custom_insights(df)
            st.write(insights)
    else:
        st.info("Please upload a dataset to get started.")

# Run the app
if __name__ == "__main__":
    main()