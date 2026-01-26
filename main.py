import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess import preprocess_genre, get_image_url

st.set_page_config(page_title="Anime Recommendation", layout="wide")

st.title("🎌 Anime Recommendation System (Genre Based)")

# =====================
# LOAD DATA
# =====================

@st.cache_data
def load_data():
    anime_genre = pd.read_csv("data/anime_recommendation.csv")
    anime_img = pd.read_csv("data/mal_anime.csv")
    return anime_genre, anime_img

anime_genre, anime_img = load_data()

# =====================
# PREPROCESS
# =====================

genre_matrix, mlb = preprocess_genre(anime_genre)

anime_img['image_url'] = anime_img['image'].apply(get_image_url)

# =====================
# SIDEBAR
# =====================

st.sidebar.header("🎯 Pilih Anime")
anime_title = st.sidebar.selectbox(
    "Pilih Anime Favorit",
    anime_genre['title'].unique()
)

# =====================
# RECOMMENDATION
# =====================

selected_index = anime_genre[anime_genre['title'] == anime_title].index[0]

similarity = cosine_similarity(
    [genre_matrix[selected_index]],
    genre_matrix
)

anime_genre['similarity'] = similarity[0]

recommendations = anime_genre.sort_values(
    by='similarity',
    ascending=False
).iloc[1:7]

# =====================
# DISPLAY
# =====================

st.subheader("✨ Rekomendasi Anime Berdasarkan Genre")

cols = st.columns(3)

for i, row in recommendations.iterrows():
    with cols[i % 3]:

        # ambil image
        img_row = anime_img[anime_img['title'] == row['title']]
        if not img_row.empty:
            img_url = get_image_url(img_row.iloc[0]['image'])
            st.image(img_url, use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x450", use_container_width=True)

        # title
        st.markdown(f"### {row['title']}")

        # genre (SUDAH LIST)
        if isinstance(row['genres'], list):
            st.write(f"Genre: {', '.join(row['genres'])}")
        else:
            st.write(f"Genre: {row['genres']}")

