# view/movie_widgets.py
import customtkinter as ctk

class MovieCard(ctk.CTkFrame):
    def __init__(self, parent, movie_id, title, poster_img, watched, on_watch_toggle):
        super().__init__(parent, width=200, height=360)
        self.grid_propagate(False)

        # Poster
        if poster_img:
            ctk.CTkLabel(self, image=poster_img, text="").pack(side="top", padx=5, pady=5)
        else:
            # If poster_img is None, place a placeholder label so layout stays consistent
            ctk.CTkLabel(self, text="", height=18).pack(side="top", padx=5, pady=5)

        # Watch/unwatch button
        btn_text = "Remove from watched list" if watched else "Click to mark as watched"
        self.watch_button = ctk.CTkButton(
            self,
            text=btn_text,
            width=150,
            command=lambda: on_watch_toggle(movie_id, self.watch_button)
        )
        self.watch_button.pack(pady=(0, 10))

        # Title label
        ctk.CTkLabel(self, text=title, font=("Arial", 14), wraplength=200, justify="center").pack(padx=5)
