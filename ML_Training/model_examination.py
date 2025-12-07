import torch
import numpy as np

# from Model_Examination.load_model import epochs
from MovieDataset import MovieDataset
from mf_model import MLP_model
from get_processed_features import generate_movie_features, generate_user_features
from recall_at_k import recall_at_k
from bpr_loss import BPRLoss
import pandas as pd
epochs = [238]
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

num_layers = params["num_layers"]
layer_dims= [params["embed_size"]] * num_layers


# def get_model_from_a_save_model(experiment_dir, epoch):

def get_model_pt(epoch, movie_feature_dim, user_features_dim,
                 movies_features, user_features, user_item_dict):

    model = MLP_model(
        num_layers=num_layers,
        layer_dims=layer_dims,
        movie_feature_size=movie_feature_dim,
        user_feature_size=user_features_dim,
        movie_emb=movies_features,
        user_emb=user_features,
        user_item_dict=user_item_dict
    )

    file = f"../experiment_runs/experiment_9/model_checkpoints/epoch_{epoch}.pt"

    checkpoint = torch.load(file, map_location="cuda:0")  # or 'cuda'
    model.load_state_dict(checkpoint)

    return model




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


user_features, movies_features = get_features()
user_features = user_features.to("cuda:0")
movies_features = movies_features.to("cuda:0")

movie_feature_dim = movies_features.size(1)
user_features_dim = user_features.size(1)

NUM_USER = user_features.size(0)
NUM_MOVIE = movies_features.size(0)

loss_f = BPRLoss()

recalls = [100]

columns = (["epoch"] +
           [f"train_k@{k}" for k in recalls] +
           [f"test_k@{k}" for k in recalls] +
           ["train_loss", "val_loss"])

results_df = pd.DataFrame(columns=columns)

for step in epochs:

    dataset = MovieDataset(NUM_USER, NUM_MOVIE,user_features,movies_features)
    user_item_dict = dataset.train_dict

    model =  get_model_pt(step, movie_feature_dim, user_features_dim,
             movies_features, user_features, user_item_dict).to("cuda:0")

    row = {"epoch": step}
    with torch.no_grad():

        tu, tp, tn = dataset.get_train_pairs()
        vu, vp, vn = dataset.get_test_pairs()

        tuser_emb, tpos_emb, tneg_emb = model(tu, tp, tn)
        vuser_emb, vpos_emb, vneg_emb = model(vu, vp, vn)

        train_loss = loss_f(tuser_emb, tpos_emb, tneg_emb)
        val_loss = loss_f(vuser_emb, vpos_emb, vneg_emb)

        row["train_loss"] = float(train_loss.item())
        row["val_loss"] = float(val_loss.item())

        for k in recalls:
            # train_recall = recall_at_k(
            #     NUM_USER, NUM_MOVIE, k, model, dataset.train_dict
            # )
            test_recall = recall_at_k(
                NUM_USER, NUM_MOVIE, k, model, dataset.test_dict, 5,dataset.train_dict,movies_features
            )

            row[f"train_k@{k}"] = float(0)
            row[f"test_k@{k}"]  = float(test_recall)

    results_df.loc[len(results_df)] = row

    print("\n========== Epoch Result ==========")
    print(pd.DataFrame([row]))
    print("==================================\n")


# ---- Save to CSV ----
# results_df.to_csv("recall_results.csv", index=False)

# print("Saved to recall_results.csv")