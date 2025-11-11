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
    def __init__(self, num_layers, layer_dims,movie_feature_size,user_feature_size):
        super().__init__()
        # num_layers , such as 4
        assert  num_layers == len(layer_dims)
        self.linear_list = nn.ModuleList([])

        for i in range(num_layers - 1):
            layer = MLP_layer(layer_dims[i], layer_dims[i + 1])
            self.linear_list.append(layer)
            print(f'created layer: {i} : {layer.linear.weight.shape}')

        self.movie_projection = nn.Linear(movie_feature_size,layer_dims[0])
        self.user_projection = nn.Linear(user_feature_size, layer_dims[0])

    def forward(self, users, pos_movies, neg_movies):
        # project movie and user features towards a common domain
        users = self.user_projection(users)
        pos_movies = self.movie_projection(pos_movies)
        neg_movies = self.movie_projection(neg_movies)
        x = torch.stack([users, pos_movies, neg_movies], dim=0)  # shape: 3 x batch x features


        for layer in self.linear_list:
            x = layer(x)
            x = F.relu(x)
            # x = F.dropout(x,p=0.25, training=self.training)

        users, pos_movies, neg_movies = x[0],x[1],x[2]

        return users, pos_movies, neg_movies
    








            


