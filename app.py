import streamlit as st
import pandas as pd
import time
from recommender.content import ContentRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.service import RecommendationService

st.title("🎬 Advanced Hybrid Movie Recommender")

# 1. Load Data with a Spinner
st.write("### System Status")
status_text = st.empty()
status_text.text("⏳ Loading movie database...")

try:
    # Attempt to load data
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    status_text.text("✅ Data loaded! Building Content Model...")
    
    # 2. Build Content Model
    content_model = ContentRecommender(movies)
    status_text.text("✅ Content Model built! Building Collaborative Model...")
    
    # 3. Build Collaborative Model
    collab_model = CollaborativeRecommender(ratings)
    status_text.text("✅ Collaborative Model built! Starting Service...")
    
    # 4. Start Service
    service = RecommendationService(movies, content_model, collab_model)
    status_text.success("🚀 System Ready!")
    time.sleep(1)
    status_text.empty() # Clear the status text

    # --- UI ---
    st.write("Select a movie to get recommendations based on **Plot** and **User Ratings**.")
    
    movie_list = movies["title"].values
    selected_movie = st.selectbox("🔍 Search for a movie:", movie_list)

    if st.button("Get Recommendations"):
        with st.spinner("Finding best matches..."):
            recommendations = service.recommend(selected_movie)
        
        if len(recommendations) > 0:
            st.write("### 🍿 Top Recommendations")
            for i, row in recommendations.iterrows():
                st.write(f"**{i+1}. {row['title']}**")
                st.caption(f"Genre: {row['genres']}")
                st.divider()
        else:
            st.error("Could not find recommendations for this movie.")

except FileNotFoundError:
    st.error("Data files not found! Please make sure 'movies.csv' and 'ratings.csv' are in the 'data' folder.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
