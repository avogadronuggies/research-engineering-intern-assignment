from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

df = pd.read_csv("../data/cleaned_data.csv")

vectorizer = CountVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(df["body"].dropna())

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

topics = lda.transform(X)
df["topic"] = topics.argmax(axis=1)

df.to_csv("../data/labeled_data.csv", index=False)
