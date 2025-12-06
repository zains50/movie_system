# matrix factorization model
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F


class MLP_layer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x)
        return x


class MLP_model(nn.Module):
    def __init__(self, num_layers, layer_dims,movie_feature_size,user_feature_size,movie_emb):
        super().__init__()
        # num_layers , such as 4
        assert  num_layers == len(layer_dims)
        self.linear_list_user = nn.ModuleList([])
        self.linear_list_movie = nn.ModuleList([])

        self.movie_emb = movie_emb


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

    def forward(self, user_emb,watched_list):
        # project movie and user features towards a common domain
        saved_user_emb = []
        saved_movie_emb = []

        user_emb = F.relu(self.user_projection(user_emb))
        movie_emb = F.relu(self.movie_projection(self.movie_emb))

        pos_movies_emb = movie_emb

        saved_user_emb.append(user_emb)
        saved_movie_emb.append(pos_movies_emb)


        for w in watched_list:
            user_emb = user_emb + movie_emb[w]

        user_emb = user_emb * 1/(len(watched_list) + 1)



        for layer in self.linear_list_user:
            user_emb = layer(user_emb)
            user_emb = F.relu(user_emb)
            user_emb = F.dropout(user_emb,training=self.training,p=0.25)
            saved_user_emb.append(user_emb)


        for layer in self.linear_list_movie:
            pos_movies_emb  = layer(pos_movies_emb)
            pos_movies_emb   = F.relu(pos_movies_emb)
            pos_movies_emb = F.dropout(pos_movies_emb,training=self.training,p=0.25)
            saved_movie_emb.append(pos_movies_emb)


        pos_movies = torch.mean(torch.stack(saved_movie_emb,dim=0),dim=0)
        user_emb = torch.mean(torch.stack(saved_user_emb,dim=0),dim=0)
        return user_emb, pos_movies












