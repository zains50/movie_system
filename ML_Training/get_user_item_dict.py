from data_extraction import get_all_features_numpy
import time
import pandas as pd


movies_arr, ratings_arr, users_arr = get_all_features_numpy()

def get_user_item_dict(users_arr=users_arr,ratings_arr=ratings_arr):
    # print(users_arr.shape) -> (6040, 5)
    # print(ratings_arr.shape) -> (1000208, 4)
    num_users = len(users_arr)
    user_item_dict = {}

    # go through all users
    for i in range(num_users):
        user_ratings = ratings_arr[ratings_arr[:,0] == i+1] # creates a list of all the movies the user has watched
        item_arr = user_ratings[:,1]
        user_item_dict[i] = item_arr-1
    return user_item_dict


