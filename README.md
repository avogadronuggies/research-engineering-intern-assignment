# research-engineering-intern-assignment

# Reddit Data Analysis Dashboard

Welcome to the **Reddit Data Analysis Dashboard**! This interactive web application allows you to analyze Reddit data, including sentiment analysis, topic modeling, word clouds, and more. The dashboard is built using Streamlit, Plotly, and other Python libraries to provide a user-friendly interface for exploring Reddit datasets.

---

## Features

1. **Sentiment Analysis**:
   - Analyze the sentiment of Reddit post titles using VADER (Valence Aware Dictionary and sEntiment Reasoner).
   - Categorize posts as positive, negative, or neutral.
   - Display top positive and negative posts.

2. **Word Cloud**:
   - Generate a word cloud from post titles to visualize the most frequent words.
   - Exclude stopwords and short words for better insights.

3. **Topic Modeling**:
   - Perform Non-Negative Matrix Factorization (NMF) to identify key topics in the dataset.
   - Display the top words for each topic and assign a dominant topic to each post.

4. **Community Analysis**:
   - Visualize the distribution of posts across top communities/accounts using a pie chart.

5. **Network Visualization**:
   - Create a network graph of accounts sharing a specific keyword, hashtag, or URL.

6. **Scatterplots**:
   - Visualize relationships between upvotes and other variables (e.g., number of comments, sentiment score).

7. **Time Series Analysis**:
   - Track the popularity of key topics over time.
   - Analyze the frequency of posts containing a specific keyword or phrase over time.

8. **Custom Insights**:
   - Generate custom insights, including sentiment distribution, top communities, and most frequent words.

---

## How to Use

1. **Upload Your Dataset**:
   - The dashboard supports JSONL and CSV file formats.
   - Upload your dataset using the sidebar.

2. **Explore the Dashboard**:
   - Once the dataset is uploaded, the dashboard will automatically generate visualizations and insights.
   - Use the interactive features to filter, search, and explore the data.

3. **Save and Share**:
   - Take screenshots of the visualizations or download the generated insights for further analysis.

---

## Screenshots

Here are some example screenshots of the dashboard in action:

### 1. Sentiment Analysis
![Sentiment Analysis](images/sentiment_analysis.png)

### 2. Word Cloud
![Word Cloud](images/wordcloud.png)

### 3. Topic Modeling
![Topic Modeling](images/topic_modeling.png)

### 4. Community Pie Chart
![Community Pie Chart](images/community_pie_chart.png)

### 5. Network Visualization
![Network Visualization](images/network_visualization.png)

### 6. Scatterplot: Upvotes vs Number of Comments
![Scatterplot](images/scatterplot.png)

### 7. Time Series of Key Topics
![Time Series](images/time_series.png)

---

## Installation

To run this dashboard locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reddit-data-analysis-dashboard.git
   cd reddit-data-analysis-dashboard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`.

---

## Code Overview

### 1. Loading the Dataset
```python
def load_dataset(file):
    if file.name.endswith('.jsonl'):
        data = [json.loads(line) for line in file]
        df = pd.DataFrame([post['data'] for post in data])
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        st.error("Unsupported file format. Please upload a JSONL or CSV file.")
        return None
    return df
```

### 2. Sentiment Analysis
```python
def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['text'] = df[['title', 'selftext']].fillna('').agg(' '.join, axis=1)
    df['sentiment_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )
    return df
```

### 3. Word Cloud Generation
```python
def generate_wordcloud(df):
    all_titles = ' '.join(df['title'])
    filtered_words = preprocess_text(all_titles)
    filtered_text = ' '.join(filtered_words)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(filtered_text)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return img
```

### 4. Topic Modeling
```python
def perform_topic_modeling(df, num_topics=5):
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
```

---

## Dependencies

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- WordCloud
- NLTK
- Scikit-learn
- VADER Sentiment
- NetworkX
- PyVis

---

## Contributing

@avogadronuggies

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


Enjoy exploring your Reddit data with this dashboard! 🚀

