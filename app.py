import os
import pickle

import requests
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── TMDB API key ──────────────────────────────────────────────────────────────
# Set your own key:
#   Option 1 (local): export TMDB_API_KEY="your_key"  then run streamlit
#   Option 2 (Streamlit Cloud): add TMDB_API_KEY in the Secrets manager
#   Get a free key at https://www.themoviedb.org/settings/api
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    """Return the full TMDB poster URL for a given movie_id."""
    if not TMDB_API_KEY:
        return "https://via.placeholder.com/500x750?text=No+API+Key"
    try:
        url = (
            f"https://api.themoviedb.org/3/movie/{movie_id}"
            f"?api_key={TMDB_API_KEY}&language=en-US"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        poster_path = resp.json().get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=Poster+Unavailable"


@st.cache_resource(show_spinner=False)
def load_model():
    """Load pickled movie list and similarity matrix."""
    with open("model/movie_list.pkl", "rb") as f:
        movies = pickle.load(f)
    with open("model/similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    return movies, similarity


def recommend(movie_title: str, movies, similarity, top_n: int = 5):
    """Return (names, poster_urls) for the top-N recommended movies."""
    matches = movies[movies["title"] == movie_title]
    if matches.empty:
        return [], []

    idx = matches.index[0]
    distances = sorted(
        enumerate(similarity[idx]), key=lambda x: x[1], reverse=True
    )

    names, posters = [], []
    for i, _ in distances[1 : top_n + 1]:
        row = movies.iloc[i]
        names.append(row["title"])
        posters.append(fetch_poster(row["movie_id"]))
    return names, posters


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommender System")
st.caption("Content-based recommendations powered by TMDB metadata & cosine similarity.")

if not TMDB_API_KEY:
    st.warning(
        "⚠️ **TMDB_API_KEY not set.** Posters won't load. "
        "Get a free key at https://www.themoviedb.org/settings/api and set the env variable.",
        icon="🔑",
    )

# Load model (cached — only runs once per session)
with st.spinner("Loading model…"):
    movies_df, similarity_matrix = load_model()

movie_list = sorted(movies_df["title"].values)
selected_movie = st.selectbox("🔍 Type or select a movie", movie_list)

if st.button("Show Recommendations", type="primary"):
    with st.spinner("Finding similar movies…"):
        rec_names, rec_posters = recommend(selected_movie, movies_df, similarity_matrix)

    if not rec_names:
        st.error("Movie not found in dataset.")
    else:
        cols = st.columns(5)
        for col, name, poster in zip(cols, rec_names, rec_posters):
            with col:
                st.image(poster, use_column_width=True)
                st.caption(name)

st.divider()
st.markdown(
    "<small>Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)</small>",
    unsafe_allow_html=True,
)
