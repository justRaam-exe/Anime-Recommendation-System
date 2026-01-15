import pandas as pd
import joblib
from sklearn.cluster import KMeans
from utils.preprocess import clean_data, get_image_url, vectorize_descriptions

# load dataset
df = pd.read_csv("data/mal_anime.csv")

# Cleaning dataset
df = clean_data(df)

# TF-IDF
X_text, tfidf = vectorize_descriptions(df['description'])

# Traim KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_text)

# Save Models
joblib.dump(kmeans, "model/kmeans_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("TF-IDF & KMeans models have been saved")