import streamlit as st
import pandas as pd
from utils.preprocess import clean_data, get_image_url
import joblib

# Page configuration

st.set_page_config(
    page_title ="Anime Recommendation System",
    layout="wide"
)

st.title("Anime Recommendation System")
st.write("Rekomendasi anime untuk pemula berdasarkan deskripsi.")

# Load Dataset

@st.cache_data
def load_data():
    df = pd.read_csv("data/mal_anime.csv")
    return df

df = load_data()
df = clean_data(df)
df['image_url'] = df['images'].apply(get_image_url)

kmeans = joblib.load("model/kmeans_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# sidebar for user input

st.sidebar.header("Pilih Anime Favorit")

anime_title = st.sidebar.selectbox(
    "Plih Anime",
    df['title'].unique()
)

# Recommendation Logic

selected_anime = df[df['title'] == anime_title].iloc[0]

X_all = tfidf.transform(df['description'])
selected_vector = tfidf.transform([selected_anime['description']])

df['cluster'] = kmeans.predict(X_all)
selected_cluster = kmeans.predict(selected_vector)[0]

recommendations = df[
    (df['cluster'] == selected_cluster) &
    (df['title'] != anime_title)
].head(6)

# Output

st.subheader("Rekomendasi Anime Serupa")

cols = st.columns(3)

for idx, row in recommendations.iterrows():
    with cols[idx % 3]:
        st.image(row['image_url'], use_container_width=True)
        st.markdown(f"### {row['title']}")
        st.write(f"Type: {row['type']}")
        st.write(f"Episodes: {row['episodes']}")
        st.write(f"Release Year: {row['release_year']}")
        st.write("---")