import torch
import torch_scatter

source = torch.tensor([0,0,1])
target = torch.tensor([0,1,0])

messages = torch.tensor([[1],[1]])   # message per node

# Convert node messages to edge messages
edge_messages = messages[source]     # (num_edges, feat_dim)

out = torch_scatter.scatter_add(
    src=edge_messages,
    index=target,
    dim=0,
    dim_size=2   # number of nodes
)

print(out)
