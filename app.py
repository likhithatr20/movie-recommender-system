# app.py
import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender — (uses movies_dict1.pkl)")

# ----- helper: fetch poster from TMDB -----
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # you can replace with your secret later

def fetch_poster(movie_id):
    try:
        if not movie_id:
            raise ValueError("no movie_id")
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        pass
    # fallback image if anything fails
    return "https://via.placeholder.com/500x750?text=No+Image"

# ----- load movies_dict1.pkl -----
@st.cache_data(show_spinner=False)
def load_movies(path="movies_dict1.pkl"):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # if the object is a dict of lists or list of dicts, convert to DataFrame
        if isinstance(obj, dict):
            movies_df = pd.DataFrame(obj)
        elif isinstance(obj, list):
            movies_df = pd.DataFrame(obj)
        elif isinstance(obj, pd.DataFrame):
            movies_df = obj
        else:
            # try to coerce
            movies_df = pd.DataFrame(obj)
        return movies_df
    except FileNotFoundError:
        st.error(f"File not found: {path}. Make sure movies_dict1.pkl is in the repo root.")
        st.stop()
    except Exception as e:
        st.error("Error loading movies_dict1.pkl")
        st.exception(e)
        st.stop()

movies = load_movies()

st.write(f"Loaded {len(movies)} movies (showing first 5 rows):")
st.dataframe(movies.head())

# fill missing expected text columns
for col in ['title','genres','overview','keywords','cast','crew','movie_id']:
    if col not in movies.columns:
        movies[col] = ""

# ----- prepare combined text and similarity -----
@st.cache_data(show_spinner=False)
def build_similarity(df):
    df = df.copy()
    df['combined'] = (df['genres'].fillna('') + " " +
                      df['overview'].fillna('') + " " +
                      df['keywords'].fillna('') + " " +
                      df['cast'].fillna('') + " " +
                      df['crew'].fillna(''))
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['combined'])
    sim = cosine_similarity(vectors)
    return sim

with st.spinner("Computing similarity (first run may take a few seconds)..."):
    similarity = build_similarity(movies)

# ----- recommendation function -----
def recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        st.warning("Selected movie not in dataset")
        return [], []
    idx = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    results = scores[1: top_n+1]  # skip the same movie
    names = []
    posters = []
    for i, _ in results:
        names.append(movies.iloc[i]['title'])
        movie_id = movies.iloc[i].get('movie_id', None)
        try:
            # movie_id could be numeric or string; handle gracefully
            posters.append(fetch_poster(int(movie_id)) if movie_id else fetch_poster(None))
        except Exception:
            posters.append(fetch_poster(None))
    return names, posters

# ----- Streamlit UI -----
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
