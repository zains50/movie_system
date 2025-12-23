# Python file to get the movie's from the database and
import json  
import os 
import torch 
import requests 
import sentence_transformers

data_file =  "_data/movie_info_with_uuid.json"

with open(data_file, encoding="utf-8") as f:
    genres = []
    movie_information = json.load(f)
print(movie_information[0])

## CODE TO DOWNLOAD POSTERS

def download_posters(datafile, poster_file):
    for d in datafile:
        if d["Genre"] == "NOT_FOUND":
            continue
    
        image_link = d["Poster"]
        uuid_code = d["UUID"]

        if image_link == "N/A":
            continue

        try:
            response = requests.get(image_link, stream=True)

            with open (f'_posters/{uuid_code}.jpg', 'wb') as handle: 
                if not response.ok:
                    print(response)
                    continue
                
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image"):
                    print('not an image')
                    continue

                for block in response.iter_content(1024):
                    if not block:
                        break

                    handle.write(block)
                print(f'downloaded movie poster: {d["Title"], d["UUID"]}')
        except requests.exceptions.RequestException as e:
            print(f'error: {e}')

download_posters(movie_information, "")