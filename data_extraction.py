import numpy as np
import pandas as pd

DIR = "ml-1m/"
MOVIES = "movies.txt"
RATINGS = "rate.txt"
USERS = "users.txt"

def get_all_features_numpy():
    movies_arr = pd.read_csv(DIR + MOVIES, sep="::", engine="python", encoding="latin1").to_numpy()
    ratings_arr = pd.read_csv(DIR + RATINGS,delimiter="::",engine="python").to_numpy()
    users_arr = pd.read_csv(DIR + USERS,delimiter="::",engine="python").to_numpy()
    return movies_arr, ratings_arr, users_arr



def get_movies_numpy():
    movie_arr = pd.read_csv(DIR + MOVIES,delimiter="::",engine="python")
    movies_arr = movie_arr.to_numpy()
    return movies_arr





