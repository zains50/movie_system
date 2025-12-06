# model/movie_view_model.py
import json
import os
import requests
from PIL import Image
import customtkinter as ctk

# import the class exactly as you requested (Option A)
from view.movie_structure import movie_structure as MovieStructure

MOVIES_BEFORE_PATH = "data/movie_information/movies_before_2000/movies_before_2000.json"
MOVIES_AFTER_PATH = "data/movie_information/movies_after_2000/movies_after_2000.json"
POSTER_DIR = "data/posters"
PLACEHOLDER_PATH = "data/assets/notavailableplaceholder.png"


class MovieViewModel:
    def __init__(self):
        self.movies = []           # list[MovieStructure]
        self.poster_cache = {}     # movie_id -> CTkImage
        os.makedirs(POSTER_DIR, exist_ok=True)

        self._load_all_movies()
        self._download_all_posters()

    # --------------------- loading ---------------------
    def _load_all_movies(self):
        movie_id = 0
        # iterate both JSON files; first = before 2000, second = after
        for path, before_2000 in [(MOVIES_BEFORE_PATH, True),
                                  (MOVIES_AFTER_PATH, False)]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except FileNotFoundError:
                # If file missing, skip gracefully
                print(f"Warning: movie data file not found: {path}")
                continue

            for m in json_data:
                # use .get with defaults so missing keys won't crash the loader
                title = m.get("Title", f"Movie {movie_id}")
                year = m.get("Year", "N/A")
                genres_raw = m.get("Genre", "")
                genres = genres_raw.split(", ") if genres_raw else []
                poster_link = m.get("Poster", None)
                summary = m.get("summary", m.get("Plot", ""))  # tolerate alternate key names

                movie = MovieStructure(
                    title=title,
                    year=year,
                    genres=genres,
                    poster_link=poster_link,
                    summary=summary,
                    before_2000=before_2000,
                    id=movie_id
                )

                self.movies.append(movie)
                movie_id += 1

    # --------------------- downloading posters ---------------------
    def _download_all_posters(self):
        for movie in self.movies:
            poster_url = movie.poster_link
            filename = f"{movie.id}.jpg"
            filepath = os.path.join(POSTER_DIR, filename)

            # if already present, attach path and continue
            if os.path.exists(filepath):
                movie.set_poster_path(filepath)
                continue

            # nothing to download
            if not poster_url or poster_url in ("N/A", None):
                continue

            # try:
            #     resp = requests.get(poster_url, timeout=5)
            #     if resp.status_code == 200 and resp.content:
            #         with open(filepath, "wb") as f:
            #             f.write(resp.content)
            #         movie.set_poster_path(filepath)
            #         # no printing in production, but useful for debugging
            #         print(f"Saved poster: {filepath}")
            # except Exception:
            #     # skip failures silently (network/timeout)
            #     continue

    # --------------------- public helpers ---------------------
    def get_poster_image(self, movie_id):
        """
        Returns a CTkImage for the given movie id. Uses the downloaded poster
        path (movie.poster_image_path) or PLACEHOLDER_PATH if none.
        Caches CTkImage objects by movie_id.
        """
        if movie_id in self.poster_cache:
            return self.poster_cache[movie_id]

        # guard against invalid ids
        if movie_id < 0 or movie_id >= len(self.movies):
            # return placeholder as a fallback
            path = PLACEHOLDER_PATH
        else:
            movie = self.movies[movie_id]
            path = movie.poster_image_path or PLACEHOLDER_PATH

        # create image and cache it
        try:
            img = Image.open(path).resize((200, 300))
            poster_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 300))
            self.poster_cache[movie_id] = poster_img
            return poster_img
        except Exception:
            # if PIL fails for any reason, return None
            return None

    def movie_list(self):
        """
        Return a list of tuples [(id, title, poster_path), ...] for all movies.
        Call as self.model.movie_list()
        """
        return [(m.id, m.title, m.poster_image_path) for m in self.movies]
