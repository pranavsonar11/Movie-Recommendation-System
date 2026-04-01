# 🎬 Movie Recommender System

A **content-based** movie recommender built with Python, scikit-learn, and Streamlit.  
Given a movie title, it returns the 5 most similar movies using cosine similarity on a combined tag vector (overview · genres · keywords · top cast · director).

---

## Demo

> Run locally — see setup below. Streamlit Cloud deployment instructions at the bottom.

---

## How It Works

1. **Feature Engineering** — Each movie's metadata (genres, keywords, cast, crew, overview) is merged into a single `tags` string.  
2. **Vectorisation** — A `CountVectorizer` (top 5 000 tokens, English stop-words removed) converts tags to a bag-of-words matrix.  
3. **Similarity** — Pairwise cosine similarity is computed across all ~4 800 movies.  
4. **Recommendation** — For any queried movie, the top-5 highest-similarity neighbours are returned with TMDB poster images.

---

## Project Structure

```
movie-recommender/
├── app.py                   # Streamlit web app
├── movie_recommender.ipynb  # Data processing + model training notebook
├── requirements.txt
├── .gitignore
├── model/                   # Auto-created by notebook (gitignored)
│   ├── movie_list.pkl       # Processed DataFrame
│   └── similarity.pkl       # ~95 MB cosine matrix
└── data/                    # Raw CSVs from Kaggle (gitignored)
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

### 2. Download the dataset

Download from Kaggle: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)  
Place both CSVs in the `data/` folder:
```
data/tmdb_5000_movies.csv
data/tmdb_5000_credits.csv
```

### 3. Generate model artefacts

Run all cells in `movie_recommender.ipynb`. This creates `model/movie_list.pkl` and `model/similarity.pkl`.

### 4. Get a TMDB API key

Register for a free API key at [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

Set it as an environment variable:
```bash
export TMDB_API_KEY="your_api_key_here"
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push the repo to GitHub (model `.pkl` files are gitignored — use **Git LFS** if you want them tracked, otherwise users regenerate locally).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo.
3. In **Advanced settings → Secrets**, add:
   ```toml
   TMDB_API_KEY = "your_api_key_here"
   ```

---

## Tech Stack

| Layer | Library |
|---|---|
| Data processing | `pandas`, `ast` |
| ML / NLP | `scikit-learn` (CountVectorizer, cosine_similarity) |
| Web app | `Streamlit` |
| Poster images | TMDB API |

---

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — provided by The Movie Database (TMDB).
