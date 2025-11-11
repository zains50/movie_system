import torch
from mf_model import MLP_model
import numpy as np
from torch.optim import Adam
from get_user_movie_features import generate_movie_features, generate_user_features
from get_user_item_dict import get_user_item_dict
from bpr_loss import BPRLoss
from tqdm import  tqdm
import time
import matplotlib.pyplot as plt

from MovieDataset import MovieDataset
from torch.utils.data import DataLoader

def train():
    movies_features, user_features = generate_movie_features(), generate_user_features()
    user_item_dict = get_user_item_dict()

    movies_features = torch.from_numpy(movies_features)
    movies_features = movies_features.to(torch.float32)

    user_features = torch.from_numpy(user_features)
    user_features = user_features.to(torch.float32)

    if torch.cuda.is_available():
        movies_features = movies_features.to("cuda:0")
        user_features = user_features.to("cuda:0")

    NUM_USER = user_features.size(0)
    NUM_MOVIE = movies_features.size(0)

    movie_feature_dim = movies_features.size(1)
    user_features_dim = user_features.size(1)

    num_layers = 3
    layer_dims = [64,64,64]

    dataset = MovieDataset(NUM_USER, NUM_MOVIE,user_features,movies_features)

    train_loader = DataLoader(
        dataset,
        batch_size=1024,      # number of users per batch
        shuffle=True,       # reshuffles users each epoch
        num_workers=2,      # parallel data loading
    )


    model = MLP_model(num_layers=num_layers, layer_dims=layer_dims,movie_feature_size=movie_feature_dim,
                      user_feature_size=user_features_dim,movie_emb=movies_features,user_emb=user_features,user_item_dict=user_item_dict)

    if torch.cuda.is_available():
        model = model.to("cuda:0")

    optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_f = BPRLoss()


    epochs = 1000
    batch_size = 2048
    loss_arr = []

    for step in tqdm(range(epochs)):
        for batch in tqdm(train_loader):
            t0 = time.time()

            user_emb, pos_movie_emb, neg_movie_emb, user_id, pos_movie_id, neg_movie_id = batch
            user_emb, pos_emb, neg_emb = model(user_emb, pos_movie_emb, neg_movie_emb, user_id, pos_movie_id, neg_movie_id)
            loss = loss_f(user_emb,pos_emb,neg_emb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t3 = time.time()

            print(f'loss: {loss}')


if __name__ == "__main__":
    # Only code inside this block runs safely with Windows multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # optional, ensures spawn method
    train()
