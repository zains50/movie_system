import numpy as np

poster_embeddings = np.load("poster_embeddings.npy")
summary_embeddings = np.load("summary_embeddings.npy")

print(f'poster emb size : {poster_embeddings.shape}')
print(f'sum emb size : {summary_embeddings.shape}')