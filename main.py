import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
def load_dataset(file_path):
    """
    Load the JSON dataset and convert it into a Pandas DataFrame.
    """
    with open('data.json', 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame([post['data'] for post in data])
    return df

def explore_data(df):
    """
    Perform basic exploration of the dataset.
    """
    print("Columns in the dataset:")
    print(df.columns)

    print("\nBasic statistics:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isnull().sum())

def analyze_engagement(df):
    """
    Analyze the most upvoted and most commented posts.
    """
    # Most upvoted posts
    most_upvoted = df.sort_values('ups', ascending=False).head(10)
    print("Most Upvoted Posts:")
    print(most_upvoted[['title', 'ups', 'num_comments']])

    # Most commented posts
    most_commented = df.sort_values('num_comments', ascending=False).head(10)
    print("\nMost Commented Posts:")
    print(most_commented[['title', 'ups', 'num_comments']])

def visualize_trends(df):
    """
    Create visualizations to understand trends in the data.
    """
    # Distribution of upvotes
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ups'], bins=30, kde=True)
    plt.title('Distribution of Upvotes')
    plt.xlabel('Upvotes')
    plt.ylabel('Frequency')
    plt.show()

    # Upvotes vs. Comments
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='ups', y='num_comments', data=df)
    plt.title('Upvotes vs. Number of Comments')
    plt.xlabel('Upvotes')
    plt.ylabel('Number of Comments')
    plt.show()

    # Top 10 most upvoted posts
    top_10_upvoted = df.sort_values('ups', ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='ups', y='title', data=top_10_upvoted, palette='viridis', hue='title', legend=False)
    plt.title('Top 10 Most Upvoted Posts')
    plt.xlabel('Upvotes')
    plt.ylabel('Post Title')
    plt.show()

def analyze_content(df):
    """
    Analyze the content of the posts (e.g., word cloud, common words).
    """
    # Combine all titles into a single string
    all_titles = ' '.join(df['title'])

    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Post Titles')
    plt.show()

    # Most common words
    words = re.findall(r'\b\w+\b', all_titles.lower())
    word_counts = Counter(words)
    print("Most Common Words:")
    print(word_counts.most_common(10))

def analyze_time_trends(df):
    """
    Analyze how post activity changes over time.
    """
    # Convert 'created' (Unix timestamp) to a readable datetime format
    df['created_date'] = pd.to_datetime(df['created'], unit='s')

    # Extract year, month, and day
    df['year'] = df['created_date'].dt.year
    df['month'] = df['created_date'].dt.month
    df['day'] = df['created_date'].dt.day

    # Group by month and count the number of posts
    posts_over_time = df.groupby(['year', 'month']).size().reset_index(name='count')

    # Plot the number of posts over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=posts_over_time.index, y='count', data=posts_over_time)
    plt.title('Number of Posts Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Posts')
    plt.show()

def save_processed_data(df, output_file):
    """
    Save the processed data to a CSV file.
    """
    df.to_csv(output_file, index=False)

def main():
    # Load the dataset
    file_path = 'reddit_posts.json'
    df = load_dataset(file_path)

    # Perform basic data exploration
    explore_data(df)

    # Analyze post engagement
    analyze_engagement(df)

    # Visualize trends
    visualize_trends(df)

    # Analyze post content
    analyze_content(df)

    # Analyze time-based trends
    analyze_time_trends(df)

    # Save processed data
    save_processed_data(df, 'processed_reddit_posts.csv')

# Run the main function
if __name__ == "__main__":
    main()