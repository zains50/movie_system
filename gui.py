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
PLACEHOLDER_PATH = "assets\\notavailableplaceholder.png"
WATCHED_LIST_PATH = "watched_movies.txt"

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
        for movie in self.movies_json[:500]:
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
                path = PLACEHOLDER_PATH

            # Create the image from the path
            img = Image.open(path).resize((200,300))
            poster_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200,300))

            # cache and return the image
            self.poster_cache[title] = poster_img
            return poster_img

# APP DEFINITION
class MovieApp(ctk.CTk):
    # method to get a set of movies the user has watched
    def load_watched_movies(self):
        try:
            with open(WATCHED_LIST_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip(",")
                return set([t for t in content.split(",") if t.strip()]) # return the content of the file as a set, if the file exists
        except:
            return set() # file doesnt exist - take no content 


    # method to add movie to watched list
    def mark_movie_watched(self, title, button):
        watched = self.load_watched_movies() # get the set of watched movies
        if title not in watched: # if the movie hasnt been watched already add it to the set
            watched.add(title)

        # add the rewrite the file with the updated set of movies the user has watched
        with open(WATCHED_LIST_PATH, "w", encoding="utf-8") as f:
            f.write(",".join(watched)+",")

        # change the button to "remove from watched list"
        button.configure(text="Click to mark as unwatched", command=lambda t=title, b=button: self.remove_watched(t,b))

    # method to remove movie from watched list
    def remove_watched(self, title, button):
        watched = self.load_watched_movies() # get the set of watched movies
        if title in watched: # if the user has watched the movie, remove it from the set
            watched.remove(title)

        with open(WATCHED_LIST_PATH, "w", encoding="utf-8") as f: # rewrite the file with the updated set
            if watched:
                f.write(",".join(watched)+",")
            else:
                f.write("")

        #reconfigure the button to add the movie as watched
        button.configure(text="Click to mark as watched", command=lambda t=title, b=button: self.mark_movie_watched(t,b))

    # method to populate tab with a given list of movies
    def populate_tab_with_movies(self, parent_frame, movie_list):
        i=0
        j=0
        watched_set = self.load_watched_movies()
        for title, _ in movie_list:
            # create a frame to put the name and image into
            frame = ctk.CTkFrame(parent_frame, width=200, height=360)
            frame.grid(row=i, column=j, padx=10, pady=10, sticky="n")
            frame.grid_propagate(False)

            # get poster from cache
            poster_img = self.view_model.get_poster_image(title)

            # if the image exists put it in a label within the frame along with a button to mark it as watched
            if poster_img:
                ctk.CTkLabel(frame, image=poster_img, text="").pack(side="top", padx=10)
                if title in watched_set:
                    # if the movie hasnt been watched add it with the remove from watched button
                    watched_button = ctk.CTkButton(frame, text="Remove from watched list", width=150)
                    watched_button.pack(pady=(0, 10))
                    watched_button.configure(command=lambda t=title, b=watched_button: self.unmark_movie_watched(t, b))
                else:
                    # if the movie hasnt been watched add it with the add to watched button
                    watched_button = ctk.CTkButton(frame, text="Click to mark as watched", width=150)
                    watched_button.pack(pady=(0, 10))
                    watched_button.configure(command=lambda t=title, b=watched_button: self.mark_movie_watched(t, b))

            # pack the frame into the given parent frame
            ctk.CTkLabel(frame, text=title, font=("Arial", 14), justify="center", wraplength=200).pack()
            # increment counters for the next movie to be placed
            j+=1
            if j >= 4:
                j=0
                i+=1
            watched_button.configure(command=lambda t=title, b=watched_button: self.mark_movie_watched(t, b))

    # CREATES A LIST OF TUPLES CORRESPONDING TO MOVIES MATCHING THE SEARCH PARAMETER
    def search_movies(self, search_parameter):
        search_parameter = search_parameter.lower().strip()
        if not search_parameter:
            return self.view_model.movie_list[:500]
        return [(title,poster) for title, poster in self.view_model.movie_list if search_parameter in title.lower()]
    
    # 
    def on_search(self, event=None):
        search_parameter = self.searchbar.get()
        matches = self.search_movies(search_parameter)

        for i in self.movies_display.winfo_children():
            i.destroy()
        
        self.populate_tab_with_movies(self.movies_display, matches)

    def __init__(self):
        super().__init__()
        self.view_model = MovieViewModel()
        self.title("Movie Recommendation System")
        self.geometry("1000x700")

        # tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True)

        home = self.tabs.add("Home")
        user_tab = self.tabs.add("User Info")
        movies_tab = self.tabs.add("Movies")
        final_tab = self.tabs.add("Final List")
        
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
        searchFrame = ctk.CTkFrame(movies_tab,height=15)
        searchFrame.pack(side="top")

        #search icon
        searchimg = Image.open("assets\\searchicon.png")
        ctksearchimg = ctk.CTkImage(light_image=searchimg, dark_image=searchimg)
        ctk.CTkLabel(searchFrame, image=ctksearchimg, text="").pack(padx=5,side="left")

        # searchbar entry field + packing
        self.searchbar = ctk.CTkEntry(searchFrame, width=300, placeholder_text="Enter a Movie Name", placeholder_text_color="#555555")
        self.searchbar.pack(side="right")
        self.searchbar.bind("<Return>",self.on_search)

        # movie list
        self.movies_display = ctk.CTkScrollableFrame(movies_tab)
        self.movies_display.pack(fill="both", expand=True)

        self.populate_tab_with_movies(parent_frame=self.movies_display, movie_list=self.view_model.movie_list[:400])


# RUN MAIN LOOP
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    MovieApp().mainloop()
