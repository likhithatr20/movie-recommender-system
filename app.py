import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your movie data
movies_dict = pickle.load(open('movies_dict1.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

st.title(" Movie Recommender System (No Pre-Trained Model)")

# Convert tags/overview into feature vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute similarity matrix on the fly
similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in movies['title'].values:
        return ["Movie not found in database!"]
    
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True, key=lambda x: x[1]
    )[1:6]
    
    recommended_movies = [movies.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Streamlit UI
selected_movie = st.selectbox(
    "Select a movie to get recommendations:",
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.write("Recommended Movies:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")

