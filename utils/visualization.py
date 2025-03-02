import streamlit as st
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import io
from utils.text_preprocessing import preprocess_text

def generate_wordcloud(df):
    """
    Generate a word cloud from post titles.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    
    Returns:
    BytesIO: Image data for the word cloud
    """
    all_titles = ' '.join(df['title'].fillna(''))
    filtered_words = preprocess_text(all_titles)
    filtered_text = ' '.join(filtered_words)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(filtered_text)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return img

def plot_word_frequency_pie(top_words):
    """
    Plot a pie chart of the top most frequent words.
    
    Parameters:
    top_words (list): List of (word, count) tuples
    """
    # Prepare data for the pie chart
    words, counts = zip(*top_words)
    fig = px.pie(names=words, values=counts, title="Top Most Frequent Words")
    st.plotly_chart(fig)

def plot_community_pie_chart(df):
    """
    Plot a pie chart of top communities/accounts.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    """
    top_communities = df['subreddit'].value_counts().head(10)
    fig = px.pie(names=top_communities.index, values=top_communities.values, title="Top Communities/Accounts")
    st.plotly_chart(fig)

def plot_upvotes_scatter(df, y_axis):
    """
    Plot a scatterplot of upvotes vs another variable.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    y_axis (str): Column name to use for the y-axis
    """
    fig = px.scatter(df, x='ups', y=y_axis, title=f"Upvotes vs {y_axis.capitalize()}",
                     labels={'ups': 'Upvotes', y_axis: y_axis.capitalize()})
    st.plotly_chart(fig)