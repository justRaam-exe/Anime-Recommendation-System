import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_data(df):
    """
    Cleaning Dataset
    """
    df = df[
        ['title', 'description', 'images', 'type',
         'episodes', 'released_season', 'release_year']
    ]
    
    df = df.dropna(subset=['description'])
    df['episodes'] = df['episodes'].fillna(0)
    df['released_season'] = df['released_season'].fillna('Unknown')
    df['type'] = df['type'].fillna('Unknown')
    
    return df

def get_image_url(image_col):
    """
    Extract image URL from the images column
    """
    try:
        return eval(image_col)['jpg']['image_url']
    except:
        return "https://via.placeholder.com/300x450"
    
def vectorize_descriptions(text_series, vectorizer=None):
    """
    TF-IDF Vectorization
    """
    
    if vectorizer:
        return vectorizer.transform(text_series)
    
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )
    vectors = tfidf.fit_transform(text_series)
    return vectors, tfidf