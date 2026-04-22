from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://faharimran.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"]
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/recommend")
def recommend(data: dict):
    result = model.predict([data["movie"]])
    return {"recommendations": result.tolist()}
