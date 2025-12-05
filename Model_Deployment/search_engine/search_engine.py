# a simple search model using pretrained sentence encoders

from sentence_transformers import SentenceTransformer
from data_extraction import get_movies_numpy
import torch

movies_arr = get_movies_numpy()
movie_names = movies_arr[:,1]
print(movie_names)
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
movie_name_encode = model.encode(movie_names,batch_size=64,show_progress_bar=True,convert_to_tensor=True)


search_query = "marvel movies"
K=20
search_query_enocde = model.encode(search_query,convert_to_tensor=True)

ranking = search_query_enocde @ movie_name_encode.T
ranking = ranking.cpu()
values, indices = torch.topk(ranking, K)
print(movie_names[indices])


