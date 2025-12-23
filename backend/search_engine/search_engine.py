import torch 
import numpy as np 
import os 
from pathlib import Path 
import pickle 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
# Code to load the search engine 
# NODE THAT THE TEXT EMBEDDING LENGTH IS 
# 768 MEANING FOR THE SEARDCH ENGINE WE CROP
class search_engine():
    def __init__(self, movie_embedding_folder):
        print("search_engine.py: LOADING SEARCH ENGINE")
        movie_embedding_folder = Path(movie_embedding_folder)
        processed_dir = Path("_data/_movie_emb_processed")
        self.TEXT_MODEL  = SentenceTransformer("all-mpnet-base-v2")
        
        if processed_dir.is_dir():
            with open (processed_dir / "uuids.pkl",  "rb") as f:
                self.uuids = pickle.load(f)
            self.list_of_movie_emb = torch.load(processed_dir / "movie_emb.pt", map_location="cpu", 
                                                weights_only=True)
        else:
            self.files = [f for f in movie_embedding_folder.iterdir() if f.is_file()]
            self.uuids = [p.stem for p in movie_embedding_folder.glob("*.npy")]
            list_of_movie_emb = []
            
            for f in tqdm(self.files):
                emb = np.load(f)
                emb = torch.tensor(emb, dtype=float)
                list_of_movie_emb.append(emb)
            self.list_of_movie_emb = torch.stack(list_of_movie_emb)[:, :768]

            processed_dir.mkdir(parents=True, exist_ok=True)
            with open(processed_dir / "uuids.pkl", "wb") as f:
                pickle.dump(self.uuids, f)
            torch.save(self.list_of_movie_emb, processed_dir / "movie_emb.pt")

        print("search_engine.py: SEARCH ENGINE LOADED")

    def search(self, search_query, return_k):
        if return_k > self.list_of_movie_emb.size(0):
            return_k = self.list_of_movie_emb.size(0)
        search_query_embed = self.TEXT_MODEL.encode(search_query)
        search_query_embed = torch.tensor(search_query_embed, dtype=float)
        search_topk = search_query_embed @ self.list_of_movie_emb.T
        _, indices = torch.topk(search_topk, return_k)

        return_values = [self.uuids[i] for i in indices]
        return return_values
    
## SINGELTON INSTANCE
SEARCH_ENGINE = search_engine("_data/_movie_emb_for_search")


# def search() RETURN : ['99023790-0811-4435-b225-5c25b966af15', 'af62396b-5c8b-4f05-b3db-418f923046b1', '72f6ae84-d222-4a34-8967-2010a81927c3', '99ca477b-c167-45a6-9e26-c8491f7e6b08', '3ee2ab82-6fe3-4c6e-bb12-3b99b67fd27b']
