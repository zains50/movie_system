from data_extraction import get_all_features_numpy
import torch
torch.set_printoptions(threshold=torch.inf)

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

def generate_movie_features(movies_arr):
    num_movies = len(movies_arr)
    num_genres = 18
    genre_map_list = []
    genres = movies_arr[:, -1]

    for genre_string in genres:
        genre_mapped = []
        genre_string = genre_string.split("|")
        for g in genre_string:
            genre_mapped.append(genres_dict[g])
        genre_map_list.append(genre_mapped)


    movie_features_empty = torch.zeros(num_movies, num_genres)

    for x in range(num_movies):
        for genres_num in genre_map_list[x]:
            movie_features_empty[x, genres_num] = 1

    return movie_features_empty

# user features will have an array with
# [Gender,Age,Occupation,m1,m2....,m_n]
# where m_n = 1 if a user has watched that movie and m_n = 0 if the user has not

# future idea: we can also include a users favoured genres

def generate_user_features(users_arr,ratings_arr):
    print(users_arr.shape)
    print(ratings_arr.shape)


generate_user_features(users_arr,ratings_arr)
