import numpy as np
import pandas as pd
from recommender.hybrid import hybrid_scores

class RecommendationService:
    def __init__(self, movies_df, content_model, collab_model):
        self.movies = movies_df.reset_index(drop=True)
        self.content_model = content_model
        self.collab_model = collab_model

    def recommend(self, title: str, k=10):
        # 1. Find movie index
        matches = self.movies[self.movies["title"] == title]
        if matches.empty:
            return []
        
        idx = matches.index[0]
        movie_id = self.movies.iloc[idx]["movieId"]

        # 2. Get scores from both models
        content_scores = self.content_model.get_scores(idx)
        collab_scores = self.collab_model.get_scores(movie_id)

        # 3. Combine scores (Hybrid)
        final_scores = hybrid_scores(content_scores, collab_scores)

        # 4. Get top K recommendations
        top_indices = np.argsort(final_scores)[::-1][1:k+1]
        
        return self.movies.iloc[top_indices][["title", "genres"]]