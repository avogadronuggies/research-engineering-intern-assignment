# **Reddit Data Analysis Dashboard**

Welcome to the **Reddit Data Analysis Dashboard**! This project is a Streamlit-based web application that allows users to analyze Reddit data interactively. It provides insights into sentiment analysis, topic modeling, network visualization, and more.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Design](#system-design)
4. [How to Use](#how-to-use)
5. [Screenshots](#screenshots)
6. [Video Demo](#video-demo)
7. [Deployment](#deployment)
8. [Contributing](#contributing)
9. [License](#license)

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
- **Topic Modeling**:
  - Extract key topics from post titles.
  - Visualize topic trends over time.
- **Network Visualization**:
  - Explore connections between authors and subreddits.
- **Interactive Filters**:
  - Filter data by date, subreddit, or sentiment.
- **Custom Insights**:
  - Generate insights using AI/ML techniques.

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

### **Thought Process**
1. **Data Loading**:
   - The app supports JSONL and CSV files for flexibility.
   - Data is loaded into a Pandas DataFrame for easy manipulation.

2. **Sentiment Analysis**:
   - VADER is used for sentiment analysis due to its effectiveness with social media text.
   - Posts are categorized as positive, negative, or neutral based on a compound score.

3. **Topic Modeling**:
   - Non-Negative Matrix Factorization (NMF) is used to identify key topics.
   - TF-IDF vectorization is applied to preprocess text data.

4. **Visualizations**:
   - Plotly is used for interactive charts (e.g., pie charts, scatterplots).
   - WordCloud generates word clouds for post titles.
   - PyVis creates interactive network graphs.

5. **Deployment**:
   - The app is deployed using Streamlit Sharing for easy access.

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

4. **Explore the Dashboard**:
   - Use the sidebar to filter data and interact with visualizations.

---

## **Screenshots**
### **1. Sentiment Analysis**
![Sentiment Analysis]()

### **2. Topic Modeling**
![Topic Modeling](/assets/topic_modeling.png)

### **3. Network Visualization**
![Network Visualization](/assets/network_visualization.png)

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

---
