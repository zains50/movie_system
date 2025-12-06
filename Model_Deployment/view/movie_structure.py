# view/movie_structure.py
class movie_structure:
    """
    Simple container for a movie. Intentionally named `movie_structure`
    per your request (Option A).
    """
    def __init__(self, title, year, genres, poster_link, summary, before_2000, id):
        self.title = title
        self.year = year
        self.genres = genres
        self.poster_link = poster_link
        self.summary = summary
        self.before_2000 = before_2000
        self.id = id

        # path on disk to downloaded poster (or None)
        self.poster_image_path = None

    def set_poster_path(self, poster_path):
        self.poster_image_path = poster_path

    def __repr__(self):
        return f"<movie_structure id={self.id} title={self.title!r}>"
