import streamlit as st
import plotly.express as px


def analyze_user_activity(df):
    """
    Analyze and visualize user activity in the dataset.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    """
    # Count posts by author
    user_post_counts = df['author'].value_counts().reset_index()
    user_post_counts.columns = ['author', 'post_count']
    
    # Get top 10 most active users
    top_users = user_post_counts.head(10)
    
    # Create bar chart
    fig = px.bar(
        top_users, 
        x='author', 
        y='post_count',
        title='Top 10 Most Active Users',
        labels={'author': 'Username', 'post_count': 'Number of Posts'},
        color='post_count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig)
    
    # Show average engagement per user
    st.subheader("User Engagement Analysis")
    
    # Group by author and calculate mean engagement metrics
    user_engagement = df.groupby('author').agg(
        avg_upvotes=('ups', 'mean'),
        avg_comments=('num_comments', 'mean'),
        avg_sentiment=('sentiment_score', 'mean'),
        total_posts=('id', 'count')
    ).reset_index()
    
    # Filter to users with at least 3 posts for more reliable statistics
    active_users = user_engagement[user_engagement['total_posts'] >= 3].sort_values('avg_upvotes', ascending=False)
    
    # Show scatter plot of average upvotes vs. average comments
    fig = px.scatter(
        active_users.head(20), 
        x='avg_upvotes', 
        y='avg_comments',
        title='User Engagement: Average Upvotes vs. Comments',
        labels={'avg_upvotes': 'Average Upvotes', 'avg_comments': 'Average Comments'},
        size='total_posts',
        color='avg_sentiment',
        hover_name='author',
        color_continuous_scale=px.colors.diverging.RdBu,
        color_continuous_midpoint=0
    )
    st.plotly_chart(fig)