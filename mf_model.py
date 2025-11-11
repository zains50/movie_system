# matrix factorization model
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from get_movie_user_features import generate_movie_features, generate_user_features
from data_extraction import get_all_features_numpy

movies_arr, ratings_arr, users_arr = get_all_features_numpy()

class MLP_layer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x)
        return x


class MLP_model(nn.Module):
    def __init__(self, num_layers, layer_dims):
        super().__init__()
        # num_layers , such as 4
        assert  num_layers == len(layer_dims)
        self.linear_list = nn.ModuleList([])

        for x in range(0,num_layers):
            input,output = layer_dims[x]
            print(input)
            layer = MLP_layer(input,output)
            self.linear_list.append(layer)
            print(f'created layer: {x} : {layer.linear.weight.shape}')

    def forward(self, x):
        for layer in self.linear_list:
            x = layer(x)
        return x
    

class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_emb, pos_emb, neg_emb):
        # user_emb, pos_emb, neg_emb have shape [batch_size, embedding_dim]

        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        diff = pos_scores - neg_scores
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-8))

        return loss

def get_items_for_user(user_id):
    usersmovies = user_features[user_id, 3:]
    movies_watched = []
    movies_not_watched = []

    for i in range(len(usersmovies)):
        if usersmovies[i] == 1:
            movies_watched.append(i)
        else:
            movies_not_watched.append(i)

    pos_idx = torch.randint(0, len(movies_watched), (1,)).item()
    neg_idx = torch.randint(0, len(movies_not_watched), (1,)).item()

    return (user_id+1, movies_watched[pos_idx]+1, movies_not_watched[neg_idx]+1)
            

user_features = generate_user_features(users_arr, ratings_arr)
movie_features = generate_movie_features(movies_arr)

model_layers = [
    (2,64),
    (64,4),
    (4,1)
]

for i in range(1000,1050):
    print(get_items_for_user(i)) # yay

bpr_loss_fn = BPRLoss()