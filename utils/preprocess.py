import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_data(df):
    """
    Cleaning Dataset
    """
    df = df[
        ['title',
         'description',
         'image',
         'Type',
         'Episodes',
         'Released_Season',
         'Released_Year']
    ]
    
    df = df.dropna(subset=['description'])
    df['Episodes'] = df['Episodes'].fillna(0)
    df['Type'] = df['Type'].fillna('Unknown')
    df['Released_Season'] = df['Released_Season'].fillna('Unknown')
    df['Released_Year'] = df['Released_Year'].fillna('Unknown')
    
    return df

def get_image_url(image_url):
    """
    Extract image URL from the images column
    """
    
    if isinstance(image_url, str) and image_url.startswith("http"):
        return image_url
    
    return "https://via.placeholder.com/300x450"
    # try:
    #     return json.loads(image_col)['jpg']['image_url']
    # except:
    #     return "https://via.placeholder.com/300x450"
    
def vectorize_descriptions(text_series):
    """
    TF-IDF Vectorization
    """
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )
    vectors = tfidf.fit_transform(text_series)
    return vectors, tfidf