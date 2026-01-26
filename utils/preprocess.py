from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_genre(df):
    df['genres'] = df['genres'].fillna("")
    df['genres'] = df['genres'].apply(lambda x: x.split(", "))

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genres'])

    return genre_matrix, mlb


def get_image_url(image_url):
    if isinstance(image_url, str) and image_url.startswith("http"):
        return image_url
    return "https://via.placeholder.com/300x450"
