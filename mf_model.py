# matrix factorization model
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch_scatter
from recall_at_k import recall_at_k
from torch.nn.utils.rnn import pad_sequence


class MLP_layer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x)
        return x


class MLP_model(nn.Module):
    def __init__(self, num_layers, layer_dims,movie_feature_size,user_feature_size,movie_emb,user_emb,user_item_dict):
        super().__init__()
        # num_layers , such as 4
        assert  num_layers == len(layer_dims)
        self.linear_list_user = nn.ModuleList([])
        self.linear_list_movie = nn.ModuleList([])

        self.movie_emb = movie_emb
        self.user_emb = user_emb
        self.user_item_dict = user_item_dict

        for i in range(num_layers - 1):
            layer = MLP_layer(layer_dims[i], layer_dims[i + 1])
            self.linear_list_user.append(layer)
            print(f'created layer: {i} : {layer.linear.weight.shape}')

        for i in range(num_layers - 1):
            layer = MLP_layer(layer_dims[i], layer_dims[i + 1])
            self.linear_list_movie.append(layer)
            print(f'created layer: {i} : {layer.linear.weight.shape}')

        self.movie_projection = nn.Linear(movie_feature_size,layer_dims[0])
        self.user_projection = nn.Linear(user_feature_size, layer_dims[0])

    def forward(self, user_ids,pos_movie_ids,neg_movie_ids):
        # project movie and user features towards a common domain

        saved_user_emb = []
        saved_movie_emb = []
        saved_negative_movie_emb = []

        user_emb = F.relu(self.user_projection(self.user_emb))
        movie_emb = F.relu(self.movie_projection(self.movie_emb))

        batch_user_emb = user_emb[user_ids]
        pos_movies_emb = movie_emb[pos_movie_ids]
        neg_movies_emb = movie_emb[neg_movie_ids]

        saved_user_emb.append(batch_user_emb)
        saved_movie_emb.append(pos_movies_emb)
        saved_negative_movie_emb.append(neg_movies_emb)

        source_list = []
        target_list = []

        for u in user_ids:
            neighbors = self.user_item_dict[int(u)]
            neighbors = [n + self.user_emb.size(0) for n in neighbors]  # shift movie IDs
            target_list.extend([u] * len(neighbors))
            source_list.extend(neighbors)

        source = torch.tensor(source_list).to(user_emb.device)
        target = torch.tensor(target_list).to(user_emb.device)

        all_item_emb = torch.cat([user_emb,movie_emb])
        edge_messages = all_item_emb[source]     # (num_edges, feat_dim)


        out = torch_scatter.scatter_mean(
            src=edge_messages,
            index=target,
            dim=0,
            dim_size=all_item_emb.size(0)
        )

        batch_user_emb = out[user_ids]

        for layer in self.linear_list_user:
            batch_user_emb = layer(batch_user_emb)
            batch_user_emb = F.relu(batch_user_emb)

            saved_user_emb.append(batch_user_emb)

        for layer in self.linear_list_movie:
            pos_movies_emb  = layer(pos_movies_emb)
            neg_movies_emb  = layer(neg_movies_emb)

            pos_movies_emb   = F.relu(pos_movies_emb)
            neg_movies_emb  = F.relu(neg_movies_emb)

            saved_movie_emb.append(pos_movies_emb)
            saved_negative_movie_emb.append(neg_movies_emb)

        users = torch.mean(torch.stack(saved_user_emb, dim=0),dim=0)
        pos_movies = torch.mean(torch.stack(saved_movie_emb,dim=0),dim=0)
        neg_movies = torch.mean(torch.stack(saved_negative_movie_emb,dim=0),dim=0)

        return users, pos_movies, neg_movies












