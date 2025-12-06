# view/movie_app_view.py
import customtkinter as ctk
from PIL import Image
from view.movie_widgets import MovieCard


# Age groups reversed
age_groups_rev = {
    "Under 18": 1,
    "18-24": 18,
    "25-34": 25,
    "35-44": 35,
    "45-49": 45,
    "50-55": 50,
    "56+": 56
}

# Occupations reversed
occupations_rev = {
    "other or not specified": 0,
    "academic/educator": 1,
    "artist": 2,
    "clerical/admin": 3,
    "college/grad student": 4,
    "customer service": 5,
    "doctor/health care": 6,
    "executive/managerial": 7,
    "farmer": 8,
    "homemaker": 9,
    "K-12 student": 10,
    "lawyer": 11,
    "programmer": 12,
    "retired": 13,
    "sales/marketing": 14,
    "scientist": 15,
    "self-employed": 16,
    "technician/engineer": 17,
    "tradesman/craftsman": 18,
    "unemployed": 19,
    "writer": 20
}

gender = {
    "male":0,
    "female":1
}


class MovieAppView(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("Movie Recommendation System")
        self.geometry("1000x700")

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True)

        self.home_tab = self.tabs.add("Home")
        self.user_tab = self.tabs.add("User Info")
        self.movies_tab = self.tabs.add("Movies")
        self.final_tab = self.tabs.add("Settings")

        self.build_home_tab()
        self.build_user_tab()
        self.build_movie_tab()
        self.build_final_tab()

    def build_home_tab(self):
        ctk.CTkLabel(
            self.home_tab,
            text="Movie Recommendation System\nZain Syed, Ryan Coones, Naufil Ansari",
            font=("Arial", 20)
        ).pack(pady=20)

    def build_final_tab(self):
        self.final_tab = self.tabs.add("Final List")

        # ===== HYPERPARAM INPUT =====
        self.hyper_label = ctk.CTkLabel(
            self.final_tab,
            text="Hyperparam (0–1):",
            font=("Arial", 16)
        )
        self.hyper_label.pack(pady=5)

        self.hyper_entry = ctk.CTkEntry(
            self.final_tab,
            placeholder_text="0.75",
            width=120
        )
        self.hyper_entry.pack(pady=5)

        # ===== ONLY AFTER 2000 CHECKBOX =====
        self.after2000_flag = ctk.BooleanVar(value=False)

        after2000_checkbox = ctk.CTkCheckBox(
            self.final_tab,
            text="Only Recommend Movies After 2000",
            variable=self.after2000_flag
        )
        after2000_checkbox.pack(pady=10)


    def get_settings(self):
        """
        Returns:
            hyperparam (float)
            only_after_2000 (bool)
        """

        # Read the entry
        text_value = self.hyper_entry.get().strip()

        # If empty, default to 0.75
        if text_value == "":
            hyper = 0.75
        else:
            try:
                hyper = float(text_value)
            except ValueError:
                hyper = 0.75  # fallback if user types nonsense

        # clamp 0–1 just in case
        hyper = max(0.0, min(1.0, hyper))

        only_after_2000 = self.after2000_flag.get()

        return hyper, only_after_2000



    def build_user_tab(self):
        # Age
        ctk.CTkLabel(self.user_tab, text="Age").pack(pady=5)
        # Use age groups keys as menu values
        self.age_menu = ctk.CTkOptionMenu(self.user_tab, values=list(age_groups_rev.keys()))
        self.age_menu.pack(pady=5)

        # Gender
        ctk.CTkLabel(self.user_tab, text="Gender").pack(pady=5)
        self.gender_menu = ctk.CTkOptionMenu(self.user_tab, values=list(gender.keys()))
        self.gender_menu.pack(pady=5)

        # Occupation
        ctk.CTkLabel(self.user_tab, text="Occupation").pack(pady=5)
        self.occupation_menu = ctk.CTkOptionMenu(self.user_tab, values=list(occupations_rev.keys()))
        self.occupation_menu.pack(pady=5)

    def get_user_numeric_info(self):
        """
        Reads the selected values from the OptionMenus and maps them to numbers.
        Returns:
            age_num (int)
            gender_num (int)
            occupation_num (int)
        """
        # Get the selected strings
        age_str = self.age_menu.get()
        gender_str = self.gender_menu.get().lower()  # make lowercase to match dictionary keys
        occupation_str = self.occupation_menu.get()

        # Map to numbers using your dictionaries
        age_num = age_groups_rev.get(age_str)
        gender_num = gender.get(gender_str)
        occupation_num = occupations_rev.get(occupation_str)

        return age_num, gender_num, occupation_num

    def build_movie_tab(self):
        # Create a frame to hold both buttons side-by-side
        button_frame = ctk.CTkFrame(self.movies_tab)
        button_frame.pack(side="top", anchor="n", pady=10)

        refresh_btn = ctk.CTkButton(
            button_frame,
            text="Refresh",
            command=self.controller.on_refresh
        )

        watch_list_refresh = ctk.CTkButton(
            button_frame,
            text="Clear Watch List",
            command=self.controller.clear_WL
        )

        # Pack them next to each other
        refresh_btn.pack(side="left", padx=5)
        watch_list_refresh.pack(side="left", padx=5)


        search_frame = ctk.CTkFrame(self.movies_tab)
        search_frame.pack(side="top", fill="x", pady=(8,4), padx=8)

        # search icon (safe fallback if file missing)
        try:
            img = Image.open("data/assets/searchicon.png")
            icon = ctk.CTkImage(light_image=img, dark_image=img, size=(20,20))
            ctk.CTkLabel(search_frame, image=icon, text="").pack(side="left", padx=5)
        except Exception:
            ctk.CTkLabel(search_frame, text="🔍").pack(side="left", padx=5)

        self.searchbar = ctk.CTkEntry(search_frame, width=300, placeholder_text="Enter a Movie Name")
        self.searchbar.pack(side="right")
        self.searchbar.bind("<Return>", lambda e: self.controller.on_search())

        self.movies_display = ctk.CTkScrollableFrame(self.movies_tab)
        self.movies_display.pack(fill="both", expand=True, padx=8, pady=8)

    def render_movies(self, movies, watched_set):
        """
        movies: list of (movie_id, title, poster_path)
        watched_set: set of movie ids (strings or ints) that are watched
        """
        # clear existing widgets
        for widget in self.movies_display.winfo_children():
            widget.destroy()

        i = j = 0
        for movie_id, title, poster_path in movies:
            poster_img = self.controller.model.get_poster_image(movie_id)
            card = MovieCard(
                self.movies_display,
                movie_id,
                title,
                poster_img,
                watched=(str(movie_id) in watched_set) or (movie_id in watched_set),
                on_watch_toggle=self.controller.toggle_watched
            )
            card.grid(row=i, column=j, padx=10, pady=10)
            j += 1
            if j >= 4:
                j = 0
                i += 1

    def render_movies_rec(self, movies, watched_set):
        """
        movies: list of (movie_id, title, poster_path)
        watched_set: set of movie ids (strings or ints) that are watched
        """
        # clear existing widgets
        for widget in self.final_tab.winfo_children():
            widget.destroy()

        i = j = 0
        for movie_id, title, poster_path in movies:
            poster_img = self.controller.model.get_poster_image(movie_id)
            card = MovieCard(
                self.movies_display,
                movie_id,
                title,
                poster_img,
                watched=(str(movie_id) in watched_set) or (movie_id in watched_set),
                on_watch_toggle=self.controller.toggle_watched
            )
            card.grid(row=i, column=j, padx=10, pady=10)
            j += 1
            if j >= 4:
                j = 0
                i += 1
