# app.py  (no sklearn required)
import streamlit as st
import pickle
import pandas as pd
import requests
import os

st.set_page_config(page_title="Movie Recommender (no sklearn)", layout="wide")
st.title("🎬 Movie Recommender (uses movies_dict1.pkl — no scikit-learn)")

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

def fetch_poster(movie_id):
    try:
        if not movie_id:
            return "https://via.placeholder.com/500x750?text=No+Image"
        url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

@st.cache_data(show_spinner=False)
def load_movies(path="movies_dict1.pkl"):
    if not os.path.exists(path):
        st.error(f"File not found: {path}. Make sure movies_dict1.pkl is in the repo root.")
        st.stop()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, dict):
        df = pd.DataFrame(obj)
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    else:
        df = pd.DataFrame(obj)
    return df

movies = load_movies()
st.write(f"Loaded {len(movies)} movies.")
st.dataframe(movies.head())

for col in ['title','genres','overview','keywords','cast','crew','movie_id']:
    if col not in movies.columns:
        movies[col] = ""

STOPWORDS = set(['the','a','an','and','or','of','in','on','with','to','for','by','from','is','are','was','were','it','this','that','as','at','be','has','have','had','but','not','its'])

def tokenize(text):
    if not isinstance(text, str):
        return set()
    text = text.lower()
    for ch in ['/', '-', '_', '.', ',', ':', ';', '(', ')', '[', ']','"','\'','?','!','&']:
        text = text.replace(ch, ' ')
    toks = [t.strip() for t in text.split() if t.strip() and t not in STOPWORDS]
    return set(toks)

@st.cache_data(show_spinner=False)
def build_token_sets(df):
    combined = (df['genres'].fillna('') + " " +
                df['overview'].fillna('') + " " +
                df['keywords'].fillna('') + " " +
                df['cast'].fillna('') + " " +
                df['crew'].fillna(''))
    token_sets = [tokenize(text) for text in combined]
    return token_sets

with st.spinner("Preparing token sets..."):
    token_sets = build_token_sets(movies)

def jaccard(a:set, b:set):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        st.warning("Selected movie not in dataset.")
        return [], []
    idx = int(movies[movies['title'] == movie_title].index[0])
    base_set = token_sets[idx]
    scores = []
    for i, s in enumerate(token_sets):
        if i == idx:
            continue
        score = jaccard(base_set, s)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_n]
    names = [movies.iloc[i]['title'] for i, _ in top]
    posters = []
    for i, _ in top:
        movie_id = movies.iloc[i].get('movie_id', None)
        try:
            posters.append(fetch_poster(int(movie_id)) if movie_id else fetch_poster(None))
        except Exception:
            posters.append(fetch_poster(None))
    return names, posters

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie:", movie_list)

if st.button("Show Recommendations"):
    names, posters = recommend(selected_movie)
    if not names:
        st.info("No recommendations found.")
    else:
        cols = st.columns(len(names))
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.text(name)
                st.image(poster, use_column_width=True)
