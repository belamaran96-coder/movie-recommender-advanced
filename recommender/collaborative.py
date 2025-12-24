import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self, ratings_df: pd.DataFrame):
        # Create user-item matrix
        user_movie_matrix = ratings_df.pivot_table(
            index="movieId",
            columns="userId",
            values="rating"
        ).fillna(0)
        
        self.movie_ids = user_movie_matrix.index.tolist()
        self.matrix = user_movie_matrix
        
        # Compute Item-Item Similarity
        self.similarity = cosine_similarity(self.matrix)

    def get_scores(self, movie_id: int):
        """Returns collaborative scores for a specific movie ID."""
        if movie_id not in self.movie_ids:
            return None
        
        idx = self.movie_ids.index(movie_id)
        return self.similarity[idx]