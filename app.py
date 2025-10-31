import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"


st.title("Movie Recommender System")

try:
    movies_dict = pickle.load(open('movies_dict1.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
except Exception as e:
    st.error("Error loading movies_dict1.pkl — make sure it's in the same folder as app.py")
    st.stop()


required_cols = ['title', 'genres', 'overview', 'keywords', 'cast', 'crew']
for col in required_cols:
    if col not in movies.columns:
        movies[col] = ""

movies['combined'] = movies['genres'] + " " + movies['overview'] + " " + movies['keywords'] + " " + movies['cast'] + " " + movies['crew']


cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['combined'].fillna(''))
similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in movies['title'].values:
        st.error("Movie not found in database!")
        return [], []
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movie_names = []
    recommended_movie_posters = []
    
    for i in distances:
        movie_id = movies.iloc[i[0]].get('movie_id', None)
        recommended_movie_names.append(movies.iloc[i[0]].title)
        if movie_id:
            recommended_movie_posters.append(fetch_poster(movie_id))
        else:
            recommended_movie_posters.append("https://via.placeholder.com/500x750?text=No+Image")
            
    return recommended_movie_names, recommended_movie_posters

selected_movie = st.selectbox("Select a movie to get recommendations:", movies['title'].values)

if st.button('Show Recommendations'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(recommended_movie_names[i])
            st.image(recommended_movie_posters[i])
