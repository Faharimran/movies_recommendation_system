from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import pandas as pd
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

# Load movies dataframe
movies = pickle.load(open(os.path.join(base_dir, "movies.pkl"), "rb"))

# Compute similarity matrix at startup (no need for similarity.pkl)
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

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

    distances = sorted(
        list(enumerate(similarity[idx])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = [movies.iloc[i[0]].title for i in distances[1:6]]
    return {"recommendations": recommendations}
