from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://faharimran.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"]
)

base_dir = os.path.dirname(__file__)
movies = pickle.load(open(os.path.join(base_dir, "movies.pkl"), "rb"))

# ✅ Fewer features = much less memory
cv = CountVectorizer(max_features=500, stop_words="english")
vectors = cv.fit_transform(movies["tags"])  # keep sparse, don't use .toarray()

@app.get("/")
def home():
    return {"status": "Movie Recommendation API is running!"}

@app.post("/recommend")
def recommend(data: dict):
    movie_name = data["movie"]
    movie_list = movies["title"].tolist()

    if movie_name not in movie_list:
        return {"error": f"Movie '{movie_name}' not found"}

    idx = movies[movies["title"] == movie_name].index[0]

    # ✅ Compute similarity only for the requested movie (not full matrix)
    movie_vector = vectors[idx]
    scores = cosine_similarity(movie_vector, vectors).flatten()

    # Get top 5 (skip index 0 = itself)
    top_indices = np.argsort(scores)[::-1][1:6]
    recommendations = [movies.iloc[i].title for i in top_indices]

    return {"recommendations": recommendations}
