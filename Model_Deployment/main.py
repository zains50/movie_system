# main.py
import customtkinter as ctk
from model.movie_view_model import MovieViewModel
from view.movie_app_view import MovieAppView
from model.get_movie_recommendations import get_model_rec
from sentence_transformers import SentenceTransformer
import numpy as np

TEXT_MODEL  = SentenceTransformer("all-mpnet-base-v2")
TITLE_SCORES = np.load("data/all_movie_title_embeddings.npy")

class Controller:
    def __init__(self):
        self.model = MovieViewModel()
        self.view = MovieAppView(controller=self)
        self.movies_watched = []

        # initial render: first movie for demonstration
        self.view.render_movies(self.model.movie_list()[:0], set())

    # --- watched list helpers (no file storage) ---
    def toggle_watched(self, movie_id, btn):
        if movie_id in self.movies_watched:
            self.movies_watched.remove(movie_id)
            btn.configure(text="Click to mark as watched")
        else:
            self.movies_watched.append(movie_id)
            btn.configure(text="Remove from watched list")
        print("Watched movies (in-memory):", self.movies_watched)

    # --- search ---
    def on_search(self):
        term = self.view.searchbar.get().lower().strip()
        all_movies = self.model.movie_list()

        if not term:
            self.view.render_movies(all_movies, set(self.movies_watched))
            return

        term_emb = TEXT_MODEL.encode([term])
        scores = term_emb @ TITLE_SCORES.T
        scores = scores.flatten()

        k = 100
        topk_indices = np.argsort(scores)[-k:][::-1]

        movie_dict = {mid: (mid, title, poster) for mid, title, poster in all_movies}
        movies = [movie_dict[mid] for mid in topk_indices if mid in movie_dict]

        self.view.render_movies(movies, set(self.movies_watched))
    def clear_WL(self):
        self.movies_watched = []
    def on_refresh(self):
        age_num, gender_num, occupation_num = self.view.get_user_numeric_info()
        hyper, only_after_2000 = self.view.get_settings()
        rec_movie_ids = get_model_rec(age_num, gender_num, occupation_num, self.movies_watched, hyper, only_after_2000, k=100)
        rec_movie_ids = [mid for sub in rec_movie_ids for mid in (sub if isinstance(sub, list) else [sub])]

        all_movies = self.model.movie_list()
        movie_dict = {mid: (mid, title, poster) for mid, title, poster in all_movies}
        movies = [movie_dict[mid] for mid in rec_movie_ids if mid in movie_dict]

        self.view.render_movies(movies, set(self.movies_watched))

    def run(self):
        self.view.mainloop()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    Controller().run()
