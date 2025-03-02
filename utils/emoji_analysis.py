from collections import Counter
import re
import emoji
import pandas as pd
import plotly.express as px
import streamlit as st

def analyze_emojis(df):
    """
    Analyze and visualize emoji usage in posts and comments.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    """
    # Function to extract emojis from text
    def extract_emojis(text):
        if not isinstance(text, str):
            return []
        return [c for c in text if c in emoji.EMOJI_DATA]
    
    # Apply extraction to title and selftext
    df['title_emojis'] = df['title'].apply(extract_emojis)
    df['selftext_emojis'] = df['selftext'].fillna('').apply(extract_emojis)
    
    # Combine all emojis
    all_emojis = []
    for emoji_list in df['title_emojis'].tolist() + df['selftext_emojis'].tolist():
        all_emojis.extend(emoji_list)
    
    # Count emoji frequencies
    emoji_counts = Counter(all_emojis)
    
    if not emoji_counts:
        st.write("No emojis found in the dataset.")
        return
    
    # Get top emojis
    top_emojis = emoji_counts.most_common(10)
    
    # Create bar chart
    emoji_df = pd.DataFrame(top_emojis, columns=['emoji', 'count'])
    
    fig = px.bar(
        emoji_df, 
        x='emoji', 
        y='count',
        title='Top 10 Most Used Emojis',
        labels={'emoji': 'Emoji', 'count': 'Frequency'},
        color='count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig)
    
    # Posts with most emojis
    df['emoji_count'] = df['title_emojis'].apply(len) + df['selftext_emojis'].apply(len)
    
    st.subheader("Posts with Most Emojis")
    emoji_rich_posts = df[df['emoji_count'] > 0].sort_values('emoji_count', ascending=False).head(5)
    if not emoji_rich_posts.empty:
        for _, post in emoji_rich_posts.iterrows():
            st.write(f"**{post['title']}** (by u/{post['author']} in r/{post['subreddit']})")
            st.write(f"Emoji count: {post['emoji_count']}")
            st.write("---")
    else:
        st.write("No posts with emojis found.")
    
    # Emoji usage by subreddit
    st.subheader("Emoji Usage by Subreddit")
    emoji_by_subreddit = df.groupby('subreddit')['emoji_count'].sum().reset_index().sort_values('emoji_count', ascending=False).head(10)
    
    fig = px.bar(
        emoji_by_subreddit, 
        x='subreddit', 
        y='emoji_count',
        title='Top 10 Subreddits by Emoji Usage',
        labels={'subreddit': 'Subreddit', 'emoji_count': 'Total Emoji Count'},
        color='emoji_count',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig)