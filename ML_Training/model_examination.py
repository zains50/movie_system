import torch
import numpy as np

from MovieDataset import MovieDataset
from get_processed_features import generate_movie_features, generate_user_features

gpu=0

params = {
    "embed_size": 512,
    "num_layers": 2,
    "batch_size": 1048,
    "epochs": 500,
    "weight_decay": 1e-05,
    "gpu": 0,
    "use_text_data": True,
    "use_poster_data": True,
    "learning rate": 0.001
}

# def get_model_from_a_save_model(experiment_dir, epoch):



def get_features(use_text_data=True, use_poster_data=True):
    user_features = generate_user_features()
    movies_features = generate_movie_features()

    if use_text_data:
        text_features = np.load("../ML_Training/advanced_movie_features/before_2000/summary_embeddings.npy")
        movies_features = np.concatenate((movies_features, text_features), axis=1)

    if use_poster_data:
        poster_features = np.load("../ML_Training/advanced_movie_features/before_2000/poster_embeddings.npy")
        movies_features = np.concatenate((movies_features,poster_features),axis=1)

    movies_features = torch.from_numpy(movies_features)
    movies_features = movies_features.to(torch.float32)

    user_features = torch.from_numpy(user_features)
    user_features = user_features.to(torch.float32)

    if torch.cuda.is_available():
        movies_features = movies_features.to(f"cuda:{gpu}")
        user_features = user_features.to(f"cuda:{gpu}")


    return user_features, movies_features


