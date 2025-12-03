import json
import os
import re
import requests
from PIL import Image
import customtkinter as ctk

# paths for movie info
MOVIES_BEFORE_PATH = "Model_Deployment\\movie_information\\movies_before_2000\\movies_before_2000.json"
MOVIES_AFTER_PATH = "Model_Deployment\\movie_information\\movies_after_2000\\movies_after_2000.json"
POSTER_DIR = "movie_posters/"
PLACEHOLDER_PATH = "notavailableplaceholder.png"

class MovieViewModel:
    def __init__(self):
        # load json files
        with open("Model_Deployment/movie_information/movies_after_2000/movies_after_2000.json", "r", encoding="utf-8") as f1, open("Model_Deployment/movie_information/movies_before_2000/movies_before_2000.json", "r", encoding="utf-8") as f2:
            self.movies_json1 = json.load(f1)
            self.movies_json2 = json.load(f2)

        # merge lists
        self.movies_json = self.movies_json1 + self.movies_json2

        # create directory for posters
        self.poster_folder = "posters"
        os.makedirs(self.poster_folder, exist_ok=True)

        # ensure posters are downloaded
        self.download_all_posters()

        # build movie list (list of [movie-title, poster-path] for each movie in the json)
        self.movie_list = self.build_movie_list()

        # loading images was incredibly slow - create a poster cache to save posters weve already loaded
        self.poster_cache = {}

    # normalize text so we can reliably look up movie titles
    def normalize_title(self, title):
        return re.sub(r'[\\/*?:"<>|]', "", title.strip())

    # download posters from dataset
    def download_all_posters(self):
        # iterate through movie objects in the movies_json list
        for movie in self.movies_json:
            title = movie.get("Title")
            poster_url = movie.get("Poster")

            # skip if title or poster url entries are invalid
            if not title or not poster_url or poster_url in ["N/A", None]:
                continue
            
            # create path for the image file
            filename = self.normalize_title(title) + ".jpg"
            filepath = os.path.join(self.poster_folder, filename)

            # skip if already downloaded
            if os.path.exists(filepath):
                continue
            
            # try downloading the poster, set a limit on the time to connect and download
            try:
                img = requests.get(poster_url, timeout=5)
                if img.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(img.content)
                    print(f"Saved poster: {filepath}")
            except:
                pass  # skip failed downloads

    # build list of tuples containing movie title along with the path for the poster corresponding to the movie
    def build_movie_list(self):
        # store movie title + full movie dictionary
        return [(i.get("Title"), self.get_poster_path(i.get("Title"))) for i in self.movies_json]

    # method to find the path to the corresponding movie title
    def get_poster_path(self, title):
        filename = self.normalize_title(title) + ".jpg"
        path = os.path.join(self.poster_folder, filename)
        return path if os.path.exists(path) else None
    
    # method to fetch the image, given a title
    def get_poster_image(self, title):
            if title in self.poster_cache: # if we have already loaded the image, get the image from the cache
                return self.poster_cache[title]

            # create a path from the title - if the path doesnt exist, replace the image with a placeholder image
            path = self.get_poster_path(title)
            if not path or not os.path.exists(path):
                path = "notavailableplaceholder.png"

            # Create the image from the path
            img = Image.open(path).resize((200,300))
            poster_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200,300))

            # cache and return the image
            self.poster_cache[title] = poster_img
            return poster_img

# APP DEFINITION
class MovieApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.view_model = MovieViewModel()
        self.title("Movie Recommendation System")
        self.geometry("1100x700")

        # tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True)

        home = self.tabs.add("Home")
        user_tab = self.tabs.add("User Info")
        movies_tab = self.tabs.add("Movies")
        final_tab = self.tabs.add("Final List")

        # method to fill a tab with movies
        def populate_tab_with_movies(self, parent_frame, movie_list):
            for i in movie_list:
                # collect title and path to poster
                title = i[0]

                # create a frame to put the name and image into
                frame = ctk.CTkFrame(parent_frame)
                frame.pack(fill="x", padx=20, pady=10)

                # get poster from cache
                poster_img = self.view_model.get_poster_image(title)

                # if the image exists put it in a label within the frame
                if poster_img:
                    ctk.CTkLabel(frame, image=poster_img, text="").pack(side="left", padx=10)

                # put the title in a label within the same frame and pack it into the window
                ctk.CTkLabel(
                    frame,
                    text=title,
                    font=("Arial", 18, "bold"),
                    justify="left",
                    wraplength=700
                ).pack(side="left", anchor="w", padx=15)

        
        # HOME TAB
        ctk.CTkLabel(home, text="Movie Recommendation System\nZain Syed, Ryan Coones, Naufil Ansari", font=("Arial", 20)).pack(pady=20)

        description_text = (
            "A personalized movie recommendation engine built on the MovieLens 1M dataset.\n\n"
            "The system learns user preferences using Bayesian Personalized Ranking, combining "
            "interaction histories with rich movie features such as genres, metadata, text-based "
            "embeddings, and poster-derived visual representations.\n\n"
            "A multi-layer embedding model maps users and movies into a shared representation space "
            "and ranks unseen titles by predicted relevance, generating tailored recommendations "
            "that adapt to each user's viewing patterns."
        )

        ctk.CTkLabel(home, text=description_text, justify="center", font=("Arial", 14), wraplength=700).pack(padx=40, pady=10)

        # USER INFO TAB
        ctk.CTkLabel(user_tab, text="Name").pack(pady=5)
        ctk.CTkEntry(user_tab).pack(pady=5)

        ctk.CTkLabel(user_tab, text="Age").pack(pady=5)
        ctk.CTkEntry(user_tab).pack(pady=5)

        ctk.CTkLabel(user_tab, text="Gender").pack(pady=5)
        ctk.CTkOptionMenu(user_tab, values=["Male","Female"]).pack(pady=5)

        # MOVIES TAB
        scroll = ctk.CTkScrollableFrame(movies_tab)
        scroll.pack(fill="both", expand=True)
        populate_tab_with_movies(self, parent_frame=scroll, movie_list=self.view_model.movie_list[:400])
        



# RUN MAIN LOOP
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    MovieApp().mainloop()
