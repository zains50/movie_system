import tmdbsimple as tmdb
from data_extraction import get_movies_numpy
import requests
import os
import json
FILE_NAME = "movies_omdb.json"
API_KEY = "deceec6f"

def get_tmbd_poser():
    movies = get_movies_numpy()

    # Load existing data if file exists
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r", encoding="utf-8") as f:
            all_responses = json.load(f)
    else:
        all_responses = []


    for movie in movies:
        movie_title = movie[1].split("(")[0]
        movie_year  = movie[1].split("(")[-1][:-1]

        url = f"http://www.omdbapi.com/?t={movie_title}&y={movie_year}&apikey={API_KEY}"

        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            if data.get("Response") == "True":
                all_responses.append(response)

            elif data.get("Response") == "False":
                json_obj = {'Title':movie_title, 'Genre':"NOT_FOUND"}
                all_responses.append(json_obj)

        except Exception:
            print("error with something about api")


    with open(FILE_NAME, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=4, ensure_ascii=False)



get_tmbd_poser()