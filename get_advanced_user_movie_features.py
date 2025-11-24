from lxml.html.builder import EMBED

from data_extraction import  get_all_features_numpy
import json
import requests
import os
import re
import numpy as np

from sentence_transformers import  SentenceTransformer
from PIL import Image
import torch

movies_arr, ratings_arr, users_arr = get_all_features_numpy()

def get_omdb_movie_dict_list():
    print(movies_arr)
    print(movies_arr.shape)
    file = "json_file/movies_omdb_1_1.json"
    omdb_movie_dict_list = []

    # Open and load the JSON
    with open(file, "r",encoding="utf-8") as f:
        data = json.load(f)
    i=0
    for x in movies_arr:
        if x[1] == "FAKE MOVIE":
            omdb_movie_dict_list.append("")
        else:
            omdb_movie_dict_list.append(data[i])
            i+=1

    return omdb_movie_dict_list

def get_movie_title_and_summary_list():
    omdb_movie_dict_list = get_omdb_movie_dict_list()
    movie_title_summ_list = []
    for x in omdb_movie_dict_list:
        if x == "":
            movie_title_summ_list.append("")
        elif x["Genre"] == "NOT_FOUND":
            movie_title_summ_list.append(f"{x['Title']}: Not found")
        else:
            movie_title_summ_list.append(f"{x['Title']}: {x['Plot']}")
    return movie_title_summ_list

def get_movie_poster_link_list():
    omdb_movie_dict_list = get_omdb_movie_dict_list()
    poster_link_list = []
    for x in omdb_movie_dict_list:
        if x == "":
            poster_link_list.append("")
        elif x["Genre"] == "NOT_FOUND":
            poster_link_list.append(f"NOT_FOUND")
        else:
                poster_link_list.append(f"{x['Poster']}")
    return poster_link_list

def save_posters():
    omdb_movie_dict_list = get_omdb_movie_dict_list()
    poster_link_list = get_movie_poster_link_list()
    for i,p in enumerate(poster_link_list):
        if p != "" and p != "NOT_FOUND":
            url = p
            file_name = f'movie_posters/{i}_{omdb_movie_dict_list[i]["Title"]}.jpg'
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(file_name, "wb") as f:
                        print(f'downloaded movie: {i}_{omdb_movie_dict_list[i]["Title"]}')
                        f.write(response.content)
                else:
                    print(f"Failed to download: {url}, status code {response.status_code}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")


def get_movie_file_names(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files = sorted(files, key=lambda f: int(re.match(r"\d+", f).group()))

    return_files = []

    i = 0
    for f in files:
        m = re.match(r"\d+", f)       # match digits at the start
        if m:
            file_num = int(m.group())  # convert to integer
        else:
            print(f, "no number at start")

        if i == file_num:
            return_files.append(f)
            i=i+1
        else:
            # print(f'else happened')
            while i-1 != file_num:
                # print(i, file_num)
                i=i+1
                return_files.append("no file found")

        # print(i-1, f)


    return return_files

def encodeText(text_list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(text_list)
    return emb

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()   # verifies header
        return True
    except Exception:
        return False


def encodeImages(movie_poster_files):
    model = SentenceTransformer("clip-ViT-B-32")

    # get embedding dimension by encoding a dummy image
    dummy = model.encode(Image.new("RGB", (32, 32)))
    emb_dim = len(dummy)
    embeddings = []
    assert len(movie_poster_files) == 3952

    for p in movie_poster_files:
        file = "movie_posters/" + p
        if is_valid_image(file):
            img = Image.open(file).convert("RGB")
            emb = model.encode(img)
            embeddings.append(emb)
        else:
            embeddings.append(np.zeros(emb_dim))

    return np.vstack(embeddings)

movie_poster_files = get_movie_file_names("movie_posters")
# print(len(movie_poster_files))
movie_summary_files = get_movie_title_and_summary_list()


for x, y ,z in zip(movies_arr, movie_summary_files,movie_poster_files):
    print(x, y,z)


# print(movie_summary_files)
# print(movi)

# poster_encodings = encodeImages(movie_poster_files)
# summary_embeddings = encodeText(movie_summary_files)

# np.save("advanced_movie_features/before_2000/poster_embeddings.npy",poster_encodings)
# np.save("advanced_movie_features/before_2000/summary_embeddings.npy",summary_embeddings)
