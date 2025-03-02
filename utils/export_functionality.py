import base64
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit as st

def generate_csv_download_link(df, filename="reddit_data.csv"):
    """
    Generate a download link for a DataFrame as a CSV file.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to export
    filename (str): Filename for the download
    
    Returns:
    str: HTML link for downloading the CSV
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def export_data_and_charts(df):
    """
    Add export functionality for the data and charts.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    """
    st.subheader("Export Options")
    
    # Create columns for the export buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Export Raw Data:")
        # Full data export
        if st.button("Export Full Dataset"):
            st.markdown(generate_csv_download_link(df, "reddit_full_data.csv"), unsafe_allow_html=True)
        
        # Export filtered data
        if st.button("Export Filtered Dataset"):
            st.markdown(generate_csv_download_link(df, "reddit_filtered_data.csv"), unsafe_allow_html=True)
    
    with col2:
        st.write("Export Analysis Results:")
        # Export sentiment analysis
        if st.button("Export Sentiment Analysis"):
            sentiment_data = df[['title', 'author', 'subreddit', 'sentiment', 'sentiment_score', 'created_date']].copy()
            st.markdown(generate_csv_download_link(sentiment_data, "reddit_sentiment_analysis.csv"), unsafe_allow_html=True)
        
        # Export topics
        if 'dominant_topic' in df.columns:
            if st.button("Export Topic Analysis"):
                topic_data = df[['title', 'author', 'subreddit', 'dominant_topic', 'created_date']].copy()
                st.markdown(generate_csv_download_link(topic_data, "reddit_topic_analysis.csv"), unsafe_allow_html=True)