import json  
import os 
import torch 
import requests
import numpy as np 
from tqdm import tqdm
from PIL import Image 
from sentence_transformers import  SentenceTransformer

data_file =  "_data/movie_info_with_uuid.json"

with open(data_file, encoding="utf-8") as f:
    genres = []
    movie_information = json.load(f)


TEXT_MODEL  = SentenceTransformer("all-mpnet-base-v2")
IMG_MODEL = SentenceTransformer("clip-ViT-B-32")

def encodeText(text_list):
    emb = TEXT_MODEL.encode(text_list)
    return emb

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.load()   # force decode
        return True
    except Exception as e:
        print(e)

        return False


dummy = IMG_MODEL.encode(Image.new("RGB", (32, 32)))
emb_dim = len(dummy)

def encodeImage(movie_poster_file):
    # get embedding dimension by encoding a dummy image
    file =  movie_poster_file
    if is_valid_image(file):
        img = Image.open(file).convert("RGB")
        emb = IMG_MODEL.encode(img)
    else:
        emb = np.zeros(emb_dim)

    return emb

def generate_movie_emb():
    for m in tqdm(movie_information):
        title = m.get("Title", f"Movie")
        year = m.get("Year", "N/A")
        genres_raw = m.get("Genre", "")
        summary = m.get("Plot", "")
        movie_id = m.get("UUID")

        # TEXT EMBEDDING
        if genres_raw == "NOT_FOUND":
            te = encodeText(f"{title}: Not found")
        else:
            te = encodeText(f"{title}: {summary}, {genres_raw}")

        # IMAGE EMBEDDING
        ie = encodeImage(f"_posters/{movie_id}.jpg")

        combined_emb = np.concatenate([te,ie])
        print(len(te))
        # np.save(f"_movie_emb_for_search/{movie_id}.npy", combined_emb)
        

generate_movie_emb()