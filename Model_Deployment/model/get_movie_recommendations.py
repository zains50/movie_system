import torch
import numpy as np
from Model_Deployment.model.machine_learning_model  import MLP_model
device = "cpu"
# device = torch.device(f"cuda:{p['gpu']}" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

p = {
    "embed_size": 512,
    "num_layers": 2,
    "batch_size": 1048,
    "epochs": 500,
    "weight_decay": 1e-05,
    "gpu": 0,
    "learning rate": 0.001
}

movie_features = np.load("data/all_movie_embeddings.npy")
movie_features = torch.from_numpy(movie_features).to(torch.float32)
movie_features = movie_features.to(device)
movie_feature_dim = movie_features.size(1)

model = MLP_model(
    num_layers=p["num_layers"],
    layer_dims=[p["embed_size"]] * p["num_layers"],
    movie_feature_size=movie_feature_dim,
    user_feature_size=3955,
    movie_emb=movie_features,
)

checkpoint_path = "../experiment_runs/experiment_9/model_checkpoints/epoch_238.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:0"))
model.to(device)
model.eval()



# --- Complete recommendation function ---
def get_model_rec(age_num, gender_num, occupation_num, movies_watched,hyper, only_after_2000, k=100):
    # Build user feature tensor
    print(gender_num)
    print(age_num)
    print(occupation_num)
    user_features = torch.tensor([[gender_num, age_num, occupation_num]], dtype=torch.float32).to(device)
    padd = 3955 - 3


    padding_tensor = torch.zeros((1, padd), dtype=torch.float32).to(device)
    user_features_padded = torch.cat([user_features, padding_tensor], dim=1).to(device)

    for mw in movies_watched:
        if mw < 3955-4:
            user_features_padded[:, mw + 3] = 1



    with torch.no_grad():
        user_emb, movie_emb = model(user_features_padded,movies_watched)

    print(movie_emb.shape)
    model_scores = user_emb @ movie_emb.T
    model_scores = model_scores.flatten()       # (num_movies,)

    # Mask already watched movies
    model_scores = F.normalize(model_scores,p=2, dim=0)

    if only_after_2000:
        movies_before_2000 = [x for x in range(3955+250)]
    else:
        movies_before_2000 = [4075, 3882]

    model_scores[list(movies_before_2000)] = float('-inf')
    if movies_watched:
        model_scores[list(movies_watched)] = float('-inf')

    genre_emb = movie_features[:, :18]

    genres_pref = torch.zeros(18)
    for mw in movies_watched:
        genres_pref += genre_emb[mw]


    genre_emb = F.normalize(genre_emb, p=2, dim=1)        # normalize rows
    genres_pref = F.normalize(genres_pref, p=2, dim=0)

    sim = F.cosine_similarity(genre_emb, genres_pref.unsqueeze(0), dim=1)
    sim = F.normalize(sim,p=2,dim=0)

    hyperparam = hyper
    overall_scores =  model_scores * hyperparam + sim * (1-hyperparam)
    topk_values, topk_indices = torch.topk(overall_scores,k=k)



    # Return top-k recommended movie indices
    # topk_values, topk_indices = torch.topk(scores, k=k)  # topk_values = top scores, topk_indices = their indices

    # Convert to Python lists if needed
    topk_values = topk_values.tolist()
    topk_indices = topk_indices.tolist()

    # print(scores)
    print(topk_values)
    print(topk_indices)
    return topk_indices
