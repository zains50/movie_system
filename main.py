import torch
# from mf_model import MLP_model
from get_user_movie_features import generate_movie_features, generate_user_features

movies_features, user_features = generate_movie_features(), generate_user_features()

print(f'loaded movie and user features')

epochs = 100
model_layers = [
    (2,64),
    (64,64),
    (64,64)
]
