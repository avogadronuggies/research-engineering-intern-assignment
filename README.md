# **Reddit Data Analysis Dashboard**

Welcome to the **Reddit Data Analysis Dashboard**! This project is a Streamlit-based web application that allows users to analyze Reddit data interactively. It provides insights into sentiment analysis, topic modeling, network visualization, and more.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Design](#system-design)
4. [Implementation Details](#implementation-details)
5. [Data Flow](#data-flow)
6. [How to Use](#how-to-use)
7. [Screenshots](#screenshots)
8. [Video Demo](#video-demo)
9. [Deployment](#deployment)
10. [Contributing](#contributing)
11. [License](#license)

---

## **Project Overview**

This dashboard is designed to help users analyze Reddit data by providing:

- **Sentiment Analysis**: Categorize posts as positive, negative, or neutral using VADER.
- **Topic Modeling**: Identify key topics in Reddit posts using Non-Negative Matrix Factorization (NMF).
- **Network Visualization**: Visualize relationships between authors and subreddits.
- **Interactive Visualizations**: Explore data through word clouds, pie charts, scatterplots, and more.

---

## **Features**

- **Sentiment Analysis**:
  - Analyze sentiment distribution across posts.
  - View top positive and negative posts.
  - Track sentiment trends over time.
- **Topic Modeling**:
  - Extract key topics from post titles using NMF.
  - Visualize topic trends over time.
  - Identify emerging discussion themes.
- **Network Visualization**:
  - Explore connections between authors and subreddits.
  - Identify key influencers and community clusters.
- **Interactive Filters**:
  - Filter data by date, subreddit, sentiment, or upvote count.
  - Dynamically update visualizations based on filters.
- **Emoji Analysis**:
  - Track emoji usage across posts and communities.
  - Identify emotional trends through emoji usage.
- **User Activity Analysis**:
  - Identify the most active users and their engagement patterns.
  - Analyze the relationship between post frequency and popularity.
- **Text Summarization**:
  - Generate concise summaries of lengthy posts.
  - Focus on key information extraction.
- **Export Functionality**:
  - Export analysis results as CSV files.
  - Export raw or filtered datasets for further analysis.

---

## **System Design**

### **Code Structure**

```
C:.
├───app.py                # Main Streamlit app entry point
├───README.md             # Project documentation
├───instructions.md       # Additional setup instructions
├───requirements.txt      # Required dependencies
├───.gitignore            # Git ignore file
├───assets/               # Static assets (e.g., CSS files)
│   └───styles.css        # Styles for the dashboard
├───data/                 # Sample datasets
│   ├───data.jsonl        # Raw Reddit data in JSONL format
│   ├───processed_reddit_posts.csv  # Preprocessed dataset
├───lib/                  # External libraries
├───streamlit/            # Streamlit app files
├───utils/                # Utility functions
│   ├───__pycache__/      # Python cache files
│   ├───__init__.py       # Package initializer
│   ├───content_filters.py  # Functions for filtering content
│   ├───data_loader.py      # Data loading utilities
│   ├───emoji_analysis.py   # Emoji-based sentiment analysis
│   ├───export_functionality.py  # Export results as CSV/PDF
│   ├───network_analysis.py  # Analyzing user interactions
│   ├───sentiment_analysis.py  # Sentiment analysis using VADER
│   ├───text_preprocessing.py  # Text cleaning and preprocessing
│   ├───text_summarization.py  # Summarization of Reddit posts
│   ├───topic_modeling.py  # Identifying key topics using NMF
│   ├───user_activity_analysis.py  # Analyzing user engagement
│   ├───visualization.py  # Functions for data visualization
```

## **Implementation Details**

### **Key Components and Technologies**

1. **Core Framework**:

   - **Streamlit**: Powers the interactive web interface and dashboard components
   - **Pandas**: Handles data manipulation and analysis

2. **Analysis Libraries**:

   - **VADER (Valence Aware Dictionary for Sentiment Reasoning)**: Used for sentiment analysis, specialized for social media text
   - **NLTK (Natural Language Toolkit)**: Provides text preprocessing capabilities, tokenization, and stopword removal
   - **Scikit-learn**: Implements topic modeling through Non-Negative Matrix Factorization (NMF) and TF-IDF vectorization

3. **Visualization Technologies**:

   - **Plotly**: Creates interactive charts (pie charts, scatterplots, bar charts)
   - **WordCloud**: Generates visual representations of word frequency
   - **PyVis**: Produces interactive network visualizations
   - **NetworkX**: Constructs and manipulates network graphs

4. **Data Handling**:

   - Supports both JSONL and CSV data formats
   - Implements comprehensive filtering mechanisms (date, subreddit, sentiment, upvotes)
   - Provides data export capabilities

5. **Key Modules**:
   - **Sentiment Analysis**: Implements VADER to analyze emotional tone in posts
   - **Topic Modeling**: Uses NMF with TF-IDF vectorization to identify key discussion topics
   - **Network Analysis**: Visualizes connections between users and communities
   - **Text Summarization**: Employs extractive summarization techniques to condense lengthy posts
   - **User Activity Analysis**: Tracks engagement patterns and post frequency
   - **Emoji Analysis**: Identifies emotional trends through emoji usage

---

## **Data Flow**

The application follows a clear data flow pattern:

1. **Data Ingestion**:

   - User uploads JSONL or CSV data through the Streamlit interface
   - Data is loaded using `data_loader.py`, which detects file format and converts to pandas DataFrame

2. **Data Preprocessing**:

   - Raw text is cleaned using functions in `text_preprocessing.py`
   - Stopwords are removed, and text is tokenized
   - Date fields are converted to proper datetime format

3. **Analysis Pipeline**:

   - **Sentiment Analysis**:

     - Text is processed through VADER sentiment analyzer in `sentiment_analysis.py`
     - Posts are categorized as positive, negative, or neutral
     - Sentiment trends are calculated over time

   - **Topic Modeling**:

     - Text is vectorized using TF-IDF
     - NMF is applied to identify key topics
     - Posts are assigned to dominant topics

   - **Network Analysis**:
     - Author-subreddit relationships are mapped
     - Network graphs are created using NetworkX
     - Visualizations rendered using PyVis

4. **Filtering**:

   - User-selected filters in `content_filters.py` are applied to the dataset
   - Date ranges, subreddits, sentiment categories, and upvote thresholds can be applied
   - All visualizations update dynamically based on filtered data

5. **Visualization**:
   - Processed data is visualized through multiple chart types
   - Interactive elements allow user exploration
   - Results can be exported for further analysis

---

## **How to Use**

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App Locally**:

   ```bash
   streamlit run app.py
   ```

3. **Upload a Dataset**:

   - Upload a JSONL or CSV file containing Reddit data.
   - The application expects columns like 'title', 'selftext', 'author', 'subreddit', 'created', 'ups', and 'num_comments'.

4. **Explore the Dashboard**:
   - Use the sidebar to filter data by date range, subreddit, upvote count, and sentiment.
   - Navigate through different analysis tabs to explore various insights.
   - Export findings as needed for further analysis.

---

## **Screenshots**

### **1. Sentiment Analysis**

![image](https://github.com/avogadronuggies/research-engineering-intern-assignment/blob/main/assests/sentiment_analysis.png)

### **2. Topic Modeling**

![image](https://github.com/avogadronuggies/research-engineering-intern-assignment/blob/main/assests/topic_modeling.png)

### **3. Network Visualization**

![image](https://github.com/avogadronuggies/research-engineering-intern-assignment/blob/main/assests/network_visualization.png)

---

## **Video Demo**

Watch the video demo of the dashboard in action:

- [Google Drive Link](https://drive.google.com/uc?id=1GX7q219C-tMwoMTgF-IY5NQyEIZTH_cp&export=download)

---

## **Deployment**

The dashboard is hosted on **Streamlit Sharing**:

- [Live Dashboard](https://researchdashbored.streamlit.app)

---

## **Contribution**

By @avogadronuggies

## **Acknowledgments**

- **Streamlit** for the amazing framework.
- **VADER** for sentiment analysis.
- **Plotly** for interactive visualizations.
- **NLTK** for natural language processing capabilities.
- **Scikit-learn** for machine learning functionality.

---
