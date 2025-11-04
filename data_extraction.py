import numpy as np
import pandas as pd

DIR = "ml-1m/"
MOVIES = "movies.txt"
RATINGS = "rate.txt"
USERS = "users.txt"

def get_all_features_numpy():
    movies_arr = pd.read_csv(DIR + MOVIES, sep="::", engine="python", encoding="latin1")
    raitngs_arr = pd.read_csv(DIR + RATINGS,delimiter="::").to_numpy()
    users_arr = pd.read_csv(DIR + USERS,delimiter="::").to_numpy()
    return movies_arr, raitngs_arr, users_arr



def get_movies_numpy():
    movie_arr = pd.read_csv(DIR + MOVIES,delimiter="::")
    movies_arr = movie_arr.to_numpy()
    return movies_arr





