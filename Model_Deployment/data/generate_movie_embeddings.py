import json
import numpy as np
from PIL import Image
from sentence_transformers import  SentenceTransformer

from tqdm import tqdm

MOVIES_BEFORE_PATH = "movie_information/movies_before_2000/movies_before_2000.json"
MOVIES_AFTER_PATH = "movie_information/movies_after_2000/movies_after_2000.json"

TEXT_MODEL  = SentenceTransformer("all-mpnet-base-v2")

def encodeText(text_list):
    emb = TEXT_MODEL.encode(text_list)
    return emb

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()   # verifies header
        return True
    except Exception:
        return False

IMG_MODEL = SentenceTransformer("clip-ViT-B-32")

def encodeImage(movie_poster_file):

    # get embedding dimension by encoding a dummy image
    dummy = IMG_MODEL.encode(Image.new("RGB", (32, 32)))
    emb_dim = len(dummy)

    file = "movie_posters/" + movie_poster_file
    if is_valid_image(file):
        img = Image.open(file).convert("RGB")
        emb = IMG_MODEL.encode(img)
    else:
        emb = np.zeros(emb_dim)

    return emb




genres_dict = {
    "Action": 0,
    "Adventure": 1,
    "Animation": 2,
    "Children's": 3,
    "Comedy": 4,
    "Crime": 5,
    "Documentary": 6,
    "Drama": 7,
    "Fantasy": 8,
    "Film-Noir": 9,
    "Horror": 10,
    "Musical": 11,
    "Mystery": 12,
    "Romance": 13,
    "Sci-Fi": 14,
    "Thriller": 15,
    "War": 16,
    "Western": 17
}





def get_movie_dict():
    movie_id = 0
    all_embeddings = []

    # iterate both JSON files; first = before 2000, second = after

    with open(MOVIES_BEFORE_PATH, "r", encoding="utf-8") as f:
        movies_before = json.load(f)

    with open(MOVIES_AFTER_PATH, "r", encoding="utf-8") as f:
        movies_after = json.load(f)

    all_movies = movies_before + movies_after




    num_movies = len(all_movies)
    num_genres = 18

    for m in tqdm(all_movies):
        title = m.get("Title", f"Movie {movie_id}")
        year = m.get("Year", "N/A")
        genres_raw = m.get("Genre", "")
        summary = m.get("Plot", "")
        movie_features = np.zeros(num_genres)

        genres = genres_raw.split(", ") if genres_raw else []
        for g in genres:
            genre_num = genres_dict.get(g)
            if genre_num is None:
                continue
            # print("BRODIE")
            # print(g)
            movie_features[genre_num] = 1

        # Text embedding
        if m.get("Genre") == "NOT_FOUND":
            te = encodeText(f"{title}: Not found")
        else:
            te = encodeText(f"{title}: {summary}, {genres_raw}")

        # Image embedding
        # ie = encodeImage(f"{movie_id}.png")  # assuming correct path handled inside encodeImage

        # Concatenate text + image embeddings
        # combined_emb = np.concatenate([te])
        all_embeddings.append(te)
            #
        movie_id += 1

    # Convert to a single numpy array and save
    all_embeddings = np.vstack(all_embeddings)
    np.save("all_movie_title_embeddings.npy", all_embeddings)
    print(f"Saved combined embeddings: {all_embeddings.shape}")

    return movie_features

get_movie_dict()