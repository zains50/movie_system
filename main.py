import torch
from mf_model import MLP_model
import numpy as np
from torch.optim import Adam
from get_user_movie_features import generate_movie_features, generate_user_features
# from get_user_item_dict import get_user_item_dict
from bpr_loss import BPRLoss
from tqdm import  tqdm
import time
import matplotlib.pyplot as plt
from recall_at_k import  recall_at_k
from MovieDataset import MovieDataset
from torch.utils.data import DataLoader

import os
save_folder = "SAVED_RUNS"


def train(embed_size=32,num_layers=3,batch_size=2048,epochs=1000,weight_decay=1e-7,gpu=0,use_text_data=True, use_poster_data=True):
    user_features = generate_user_features()
    movies_features = generate_movie_features()

    if use_text_data:
        text_features = np.load("advanced_movie_features/before_2000/summary_embeddings.npy")
        movies_features = np.concatenate((movies_features, text_features), axis=1)

    if use_poster_data:
        poster_features = np.load("advanced_movie_features/before_2000/poster_embeddings.npy")
        movies_features = np.concatenate((movies_features,poster_features),axis=1)


    movies_features = torch.from_numpy(movies_features)
    movies_features = movies_features.to(torch.float32)

    user_features = torch.from_numpy(user_features)
    user_features = user_features.to(torch.float32)

    if torch.cuda.is_available():
        movies_features = movies_features.to(f"cuda:{gpu}")
        user_features = user_features.to(f"cuda:{gpu}")

    NUM_USER = user_features.size(0)
    NUM_MOVIE = movies_features.size(0)

    movie_feature_dim = movies_features.size(1)
    user_features_dim = user_features.size(1)

    layer_dims = [embed_size] * num_layers

    dataset = MovieDataset(NUM_USER, NUM_MOVIE,user_features,movies_features)


    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,      # number of users per batch
        shuffle=True,       # reshuffles users each epoch
        num_workers=0,      # parallel data loading
    )

    model = MLP_model(num_layers=num_layers, layer_dims=layer_dims,movie_feature_size=movie_feature_dim,
                      user_feature_size=user_features_dim,movie_emb=movies_features,user_emb=user_features,user_item_dict=dataset.train_dict)

    if torch.cuda.is_available():
        model = model.to(f"cuda:{gpu}")

    optimizer = Adam(params=model.parameters(), lr=0.001,weight_decay=weight_decay)
    loss_f = BPRLoss()


    loss_arr = []
    print(f'BEFORE TRAINING')
    recall =  recall_at_k(NUM_USER,NUM_MOVIE, 20, model, dataset.test_dict)
    print(f'RECALL: {recall}')

    for step in tqdm(range(epochs)):
        for i,batch in tqdm(enumerate(train_loader)):
            t0 = time.time()
            user_emb, pos_movie_emb, neg_movie_emb, user_id, pos_movie_id, neg_movie_id = batch
            user_emb, pos_emb, neg_emb = model(user_id, pos_movie_id, neg_movie_id)
            loss = loss_f(user_emb,pos_emb,neg_emb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t3 = time.time()

            if i % 10 == 0:
                with torch.no_grad():
                    test_recall =  recall_at_k(NUM_USER,NUM_MOVIE, 20, model, dataset.test_dict)
                    train_recall = recall_at_k(NUM_USER,NUM_MOVIE, 20, model, dataset.train_dict)
                    vu,vp,vn = dataset.get_test_pairs()
                    vuser_emb, vpos_emb, vneg_emb = model(user_id, pos_movie_id, neg_movie_id)

                    validation_loss = loss_f(vuser_emb,vpos_emb,vneg_emb)
                    print("===== Results =====")
                    print(f"Train: {train_recall:.4f}")
                    print(f"Test : {test_recall:.4f}")
                    print(f'Train Loss: {loss:.4f}')
                    print(f'Validation Loss: {validation_loss:.4f}')
                    print("=============================")



if __name__ == "__main__":
    # Only code inside this block runs safely with Windows multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # optional, ensures spawn method
    train()
