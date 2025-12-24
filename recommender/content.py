import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies = movies_df
        # Create a text field from genres (and optionally title)
        self.movies["text"] = self.movies["genres"].fillna("")
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["text"])
        
        # Compute cosine similarity matrix
        self.similarity = cosine_similarity(self.tfidf_matrix)

    def get_scores(self, movie_idx: int):
        """Returns similarity scores for a specific movie index."""
        return self.similarity[movie_idx]