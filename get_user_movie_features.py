from data_extraction import get_all_features_numpy
import torch
torch.set_printoptions(threshold=torch.inf)
import os

import numpy as np

import pandas as pd
movies_arr, ratings_arr, users_arr = get_all_features_numpy()

gender_map = {
    "F":0,
    "M":1
}

age_group_map = {
    1:  0,
    18: 1,
    25: 2,
    35: 3,
    45: 4,
    50: 5,
    56: 6
}

# occupation is already in dict format

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





# simple movie feature vector, for each movie we give them
# a 17 dimensional array where each dimension represents a certain
# genre. If a movie is tagged with a genre then x[genre] = 1, 0 otherwise.

def generate_movie_features():

    if os.path.exists("ml-1m/movie_features.npy"):
        movie_features = np.load("ml-1m/movie_features.npy")
        print(f'len mf: {len(movie_features)}')
        print(f'loading saved movie features file')

    else:
        num_movies = len(movies_arr)
        # print(f'len movies arr: {num_movies}')
        num_genres = 18
        genre_map_list = []
        genres = movies_arr[:, -1]

        for genre_string in genres:
            genre_mapped = []
            genre_string = genre_string.split("|")
            for g in genre_string:
                genre_mapped.append(genres_dict[g])
            genre_map_list.append(genre_mapped)


        movie_features = np.zeros((num_movies, num_genres))

        for x in range(num_movies):
            for genres_num in genre_map_list[x]:
                movie_features[x, genres_num] = 1

        np.save('ml-1m/movie_features.npy', movie_features)


    return movie_features

# user features will have an array with
# [Gender,Age,Occupation,m1,m2....,m_n]
# where m_n = 1 if a user has watched that movie and m_n = 0 if the user has not

# future idea: we can also include a users favoured genres

def generate_user_features():
    # print(users_arr.shape) -> (6040, 5)
    # print(ratings_arr.shape) -> (1000208, 4)

    if os.path.exists("ml-1m/user_features.npy"):
        user_features = np.load("ml-1m/user_features.npy")
        print(f'loading saved user features file')

    else:
        num_users = len(users_arr)
        user_features = np.zeros((num_users,3+3952)) # default value for each movie will be 0, and we update the indexes corresponding to the movies the user has watched.

        # convert users_arr to a dataframe (easier to work with)
        users_df = pd.DataFrame(users_arr)
        users_df.columns = ["user_id", "gender", "age", "occupation", "zip"]


        # populate the first 3 columns of the feature vectors for the users
        user_features[:,0] = users_df["gender"].map(gender_map) # first column - gender
        user_features[:,1] = users_df["age"].map(age_group_map) # second column - age group
        user_features[:,2] = users_df["occupation"] # third column - occupation

        # go through all users
        for i in range(0,num_users):
            user_ratings = ratings_arr[ratings_arr[:,0] == i+1] # creates a list of all the movies the user has watched
            for movie_id in user_ratings[:,1]: # iterates through the list of movies the user has watched
                user_features[i, movie_id + 2] = 1  # sets the corresponding index to 1, since they have watched it

        np.save('ml-1m/user_features.npy', user_features)

    
    return user_features

# print(user1.shape)
# generate_user_features(users_arr,ratings_arr)