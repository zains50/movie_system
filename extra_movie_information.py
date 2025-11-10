import tmdbsimple as tmdb
from data_extraction import get_movies_numpy
import requests
import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
FILE_NAME = "json_file/movies_omdb.json"

def get_tmbd_poser():
    movies = get_movies_numpy()

    with open(FILE_NAME, "r",encoding="utf-8") as f:
        movies_json_list = json.load(f)


    count = len(movies_json_list)
    for movie in movies[count+2:]:
        movie_title = movie[1].split("(")[0]
        movie_year  = movie[1].split("(")[-1][:-1]

        url = f"http://www.omdbapi.com/?t={movie_title}&y={movie_year}&apikey={API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            print(data)
            if data.get("Response") == "True":
                movies_json_list.append(data)

            elif data.get("Response") == "False":
                json_obj = {'Title':movie_title, 'Genre':"NOT_FOUND"}
                movies_json_list.append(json_obj)

        except Exception:
            print("error with something about api")


        with open(FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(movies_json_list, f, indent=4, ensure_ascii=False)



get_tmbd_poser()