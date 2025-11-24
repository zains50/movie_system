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
import json
import csv


import os
save_folder = "SAVED_RUNS"

def create_experiment_folder(base="experiment_runs"):
    os.makedirs(base, exist_ok=True)

    # Find next available experiment index
    existing = [
        int(x.split("_")[1])
        for x in os.listdir(base)
        if x.startswith("experiment_") and x.split("_")[1].isdigit()
    ]

    next_idx = max(existing) + 1 if existing else 1
    folder = os.path.join(base, f"experiment_{next_idx}")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "model_checkpoints"), exist_ok=True)

    return folder


def save_params_json(folder, params):
    with open(os.path.join(folder, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def open_recall_csv(folder):
    csv_path = os.path.join(folder, "recall_log.csv")
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_recall", "test_recall", "train_loss","val_loss"])
    return f, writer



def train(embed_size=64,num_layers=4,batch_size=2048,epochs=1000,weight_decay=1e-6,gpu=0,save_every=5,use_text_data=False, use_poster_data=False):

    exp_folder = create_experiment_folder()
    print(f"Experiment folder created: {exp_folder}")

    # Save experiment params
    params = {
        "embed_size": embed_size,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "gpu": gpu,
        "use_text_data": use_text_data,
        "use_poster_data": use_poster_data,
    }
    save_params_json(exp_folder, params)
    print("📝 Saved params.json")

    # Recall CSV logger
    csv_file, csv_writer = open_recall_csv(exp_folder)
    print("📄 recall_log.csv created")

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
            model.train()
            t0 = time.time()
            user_emb, pos_movie_emb, neg_movie_emb, user_id, pos_movie_id, neg_movie_id = batch
            user_emb, pos_emb, neg_emb = model(user_id, pos_movie_id, neg_movie_id)
            loss = loss_f(user_emb,pos_emb,neg_emb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t3 = time.time()

            if i % 5 == 0:
                with torch.no_grad():
                    test_recall =  recall_at_k(NUM_USER,NUM_MOVIE, 20, model, dataset.test_dict)
                    train_recall = recall_at_k(NUM_USER,NUM_MOVIE, 20, model, dataset.train_dict)
                    vu,vp,vn = dataset.get_test_pairs()
                    vuser_emb, vpos_emb, vneg_emb = model(user_id, pos_movie_id, neg_movie_id)


                    validation_loss = loss_f(vuser_emb,vpos_emb,vneg_emb)

                    csv_writer.writerow([step, train_recall, test_recall, loss,validation_loss])


                    print("===== Results =====")
                    print(f"Train: {train_recall:.4f}")
                    print(f"Test : {test_recall:.4f}")
                    print(f'Train Loss: {loss:.4f}')
                    print(f'Validation Loss: {validation_loss:.4f}')
                    print("=============================")

        # Save model checkpoint
        # -----------------------------
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(exp_folder, "model_checkpoints", f"epoch_{step+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved model checkpoint: {ckpt_path}")

    csv_file.close()


if __name__ == "__main__":
    # Only code inside this block runs safely with Windows multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # optional, ensures spawn method
    train()
