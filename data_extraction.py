import numpy as np
import pandas as pd
import os
from pathlib import Path

DIR = "ml-1m/"
MOVIES = "movies.txt"
RATINGS = "rate.txt"
USERS = "users.txt"

def get_all_features_numpy():
    pickle_path = Path("movie_arrays/arrays.pkl")
    if os.path.exists(pickle_path):
        print("Loading from pickle...")

        movies_arr,ratings_arr,users_arr = pd.read_pickle(pickle_path)
    else:
        print("Reading Raw Data")
        movies_arr = pd.read_csv(DIR + MOVIES, sep="::", engine="python", encoding="latin1").to_numpy()
        ratings_arr = pd.read_csv(DIR + RATINGS,delimiter="::",engine="python").to_numpy()
        users_arr = pd.read_csv(DIR + USERS,delimiter="::",engine="python").to_numpy()
        all_arrays = (movies_arr,ratings_arr,users_arr)
        pickle_path.parent.mkdir(parents = True, exist_ok = True)
        pd.to_pickle(all_arrays, pickle_path)
        print(f'len movie ar: {len(movies_arr)}')
    return movies_arr, ratings_arr, users_arr








