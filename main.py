from flask import Flask, render_template, jsonify
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
import io
import base64
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Download the VADER lexicon and stopwords
nltk.download('vader_lexicon')
nltk.download('stopwords')

app = Flask(__name__)

def load_dataset(file_path):
    """Load the JSON dataset and convert it into a Pandas DataFrame."""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame([post['data'] for post in data])
    return df

def preprocess_text(text):
    """Preprocess text by removing stopwords and non-alphabetic characters."""
    # Load stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove non-alphabetic characters and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stopwords
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]  # Exclude short words
    return filtered_words

def detect_fake_news(df):
    """Detect fake news based on keywords."""
    # List of suspicious keywords (can be expanded)
    suspicious_keywords = [
        'fake', 'hoax', 'conspiracy', 'misinformation', 'disinformation',
        'propaganda', 'rumor', 'unverified', 'false', 'debunked'
    ]
    
    # Check if title contains any suspicious keywords
    df['fake_news'] = df['title'].apply(lambda x: any(word in x.lower() for word in suspicious_keywords))
    return df

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
    return base64.b64encode(img.getvalue()).decode()

def analyze_sentiment(df):
    """Perform sentiment analysis on post titles using NLTK's VADER."""
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df

def generate_plots(df):
    """Generate Plotly visualizations."""
    # Distribution of upvotes
    fig1 = px.histogram(df, x='ups', nbins=30, title='Distribution of Upvotes')
    
    # Upvotes vs. Comments
    fig2 = px.scatter(df, x='ups', y='num_comments', title='Upvotes vs. Number of Comments')
    
    # Top 10 most upvoted posts
    top_10_upvoted = df.sort_values('ups', ascending=False).head(10)
    fig3 = px.bar(top_10_upvoted, x='ups', y='title', title='Top 10 Most Upvoted Posts')
    
    # Number of posts over time
    df['created_date'] = pd.to_datetime(df['created'], unit='s')
    posts_over_time = df.groupby(pd.Grouper(key='created_date', freq='M')).size().reset_index(name='count')
    fig4 = px.line(posts_over_time, x='created_date', y='count', title='Number of Posts Over Time')
    
    # Sentiment analysis
    sentiment_distribution = px.histogram(df, x='sentiment', nbins=30, title='Sentiment Distribution of Post Titles',
                                         labels={'sentiment': 'Sentiment Score', 'count': 'Frequency'})
    sentiment_over_time = px.line(df.groupby(pd.Grouper(key='created_date', freq='M'))['sentiment'].mean().reset_index(),
                                 x='created_date', y='sentiment', title='Average Sentiment Over Time',
                                 labels={'created_date': 'Date', 'sentiment': 'Average Sentiment Score'})
    
    # Fake news analysis
    fake_news_distribution = px.pie(df, names='fake_news', title='Fake News Distribution',
                                    labels={'fake_news': 'Fake News'})
    
    return fig1, fig2, fig3, fig4, sentiment_distribution, sentiment_over_time, fake_news_distribution

@app.route('/')
def dashboard():
    # Load the dataset
    df = load_dataset('./data/data.jsonl')
    
    # Perform sentiment analysis
    df = analyze_sentiment(df)
    
    # Detect fake news
    df = detect_fake_news(df)
    
    # Generate visualizations
    wordcloud_img = generate_wordcloud(df)
    fig1, fig2, fig3, fig4, sentiment_distribution, sentiment_over_time, fake_news_distribution = generate_plots(df)
    
    # Convert Plotly figures to HTML
    plot1 = fig1.to_html(full_html=False)
    plot2 = fig2.to_html(full_html=False)
    plot3 = fig3.to_html(full_html=False)
    plot4 = fig4.to_html(full_html=False)
    sentiment_plot1 = sentiment_distribution.to_html(full_html=False)
    sentiment_plot2 = sentiment_over_time.to_html(full_html=False)
    fake_news_plot = fake_news_distribution.to_html(full_html=False)
    
    # Render the dashboard template
    return render_template(
        'index.html',
        wordcloud_img=wordcloud_img,
        plot1=plot1,
        plot2=plot2,
        plot3=plot3,
        plot4=plot4,
        sentiment_plot1=sentiment_plot1,
        sentiment_plot2=sentiment_plot2,
        fake_news_plot=fake_news_plot
    )

@app.route('/top_words')
def top_words():
    """Fetch the top N words and their frequencies."""
    df = load_dataset('./data/data.jsonl')
    all_titles = ' '.join(df['title'])
    filtered_words = preprocess_text(all_titles)
    word_counts = Counter(filtered_words)
    
    # Get the top 10 most frequent words
    top_words = word_counts.most_common(10)
    
    # Prepare data for the pie chart
    labels = [word for word, count in top_words]
    values = [count for word, count in top_words]
    
    return jsonify({'labels': labels, 'values': values})

if __name__ == '__main__':
    app.run(debug=True)