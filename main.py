from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import os
import uuid
import numpy as np
from embeddings import get_text_embedding, cosine_similarity

app = FastAPI()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["lost_and_found"]
collection = db["objects"]

class SearchRequest(BaseModel):
    description: str

@app.post("/add")
async def add_object(description: str = Form(...), image: UploadFile = None):
    obj_id = str(uuid.uuid4())
    image_path = None

    if image:
        image_ext = image.filename.split(".")[-1]
        image_path = f"images/{obj_id}.{image_ext}"
        os.makedirs("images", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(await image.read())

    embedding = get_text_embedding(description).tolist()

    obj = {
        "description": description,
        "image_path": image_path,
        "desc_embedding": embedding
    }

    result = collection.insert_one(obj)
    return {"status": "ok", "id": str(result.inserted_id)}

@app.post("/search")
async def search_object(request: SearchRequest):
    query_emb = get_text_embedding(request.description)

    results = []
    for obj in collection.find():
        sim = cosine_similarity(query_emb, np.array(obj["desc_embedding"]))
        if sim > 0.7:
            results.append({
                "id": str(obj["_id"]),
                "score": float(sim),
                "description": obj["description"],
                "image_path": obj.get("image_path")
            })

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"matches": sorted_results}
