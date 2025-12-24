import streamlit as st
import pandas as pd
from recommender.content import ContentRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.service import RecommendationService

# Load Data
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

st.title("🎬 Advanced Hybrid Movie Recommender")

try:
    movies, ratings = load_data()
    
    # Initialize Models
    content_model = ContentRecommender(movies)
    collab_model = CollaborativeRecommender(ratings)
    service = RecommendationService(movies, content_model, collab_model)

    # User Interface
    selected_movie = st.selectbox("Select a movie you like:", movies["title"].values)

    if st.button("Recommend"):
        recommendations = service.recommend(selected_movie)
        
        if len(recommendations) > 0:
            st.write("### Recommended for you:")
            for _, row in recommendations.iterrows():
                st.write(f"**{row['title']}** ({row['genres']})")
        else:
            st.error("Movie not found or no recommendations available.")

except FileNotFoundError:
    st.error("Data files not found! Please make sure 'movies.csv' and 'ratings.csv' are in the 'data' folder.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
