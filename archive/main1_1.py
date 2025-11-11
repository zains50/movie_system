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




movies_features, user_features = generate_movie_features(), generate_user_features()
user_item_dict = get_user_item_dict()

print(movies_features.shape)
print(user_features.shape)
print(f'loaded movie and user features')

def get_items_for_user(user_id,NUM_MOVIES):
    movies_watched = user_item_dict[user_id]


    movies_not_watched = set(range(NUM_MOVIES-1)) - set(movies_watched)
    movies_not_watched = list(movies_not_watched)

    # for i in range(len(usersmovies)-1):
    #     # print(len(usersmovies))
    #     if usersmovies[i] == 1:
    #         movies_watched.append(i)
    #     else:
    #         movies_not_watched.append(i)

    pos_idx = torch.randint(0, len(movies_watched), (1,)).item()
    neg_idx = torch.randint(0, len(movies_not_watched), (1,)).item()

    return (user_id, movies_watched[pos_idx], movies_not_watched[neg_idx])

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
print(NUM_MOVIE)

print(movies_features.shape)

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
    for b in batch_size:
        t0 = time.time()
        user_batch = np.random.randint(0, NUM_USER, size=batch_size)
        # print(user_batch)
        triplets = [get_items_for_user(u,NUM_MOVIE) for u in user_batch]
        t1 = time.time()
        users, pos_movie, neg_movie = zip(*triplets)
        user_ids = list(users)
        pos_movie_ids= list(pos_movie)
        neg_movie_ids = list(neg_movie)

        t2 = time.time()
        user_emb, pos_emb, neg_emb = user_features[user_ids], movies_features[pos_movie_ids], movies_features[neg_movie_ids]
        user_emb, pos_emb, neg_emb = model(user_emb,pos_emb,neg_emb,user_ids,pos_movie_ids,neg_movie_ids)
        loss = loss_f(user_emb,pos_emb,neg_emb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        t3 = time.time()

        print(f'loss: {loss}')
        loss_arr.append(loss.cpu().item())
        # print(f'time to get list: {t1-t0}')
        # print(f'time to for/bac pass: {t3-t2}')
    # loss_arr = loss_arr.cpu().numpy()
    plt.plot(loss_arr)
    plt.show()