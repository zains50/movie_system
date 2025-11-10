# matrix factorization model
import torch
import torch.nn as nn

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



model_layers = [
    (2,64),
    (64,4),
    (4,1)
]








