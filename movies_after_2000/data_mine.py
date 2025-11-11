# import pandas as pd
# import os
# from dotenv import load_dotenv
# import requests
import json
# load_dotenv()
#
# API_KEY = os.getenv('API_KEY')
#
#
# df = pd.read_csv("IMDB Movies 2000 - 2020.csv")
# df_list = df[["original_title","year"]].values.tolist()
# print(df)
#
# movies_json_list = []
#
# for movie in df_list:
#     movie_title = movie[0]
#     movie_year = movie[1]
#
#     url = f"http://www.omdbapi.com/?t={movie_title}&y={movie_year}&apikey={API_KEY}"
#     try:
#         response = requests.get(url, timeout=10)
#         data = response.json()
#         print(data)
#         if data.get("Response") == "True":
#             movies_json_list.append(data)
#
#         elif data.get("Response") == "False":
#             json_obj = {'Title':movie_title, 'Genre':"NOT_FOUND",'Year':movie_year}
#             movies_json_list.append(json_obj)
#
#     except Exception:
#         print("error with something about api")
#
#
#     with open("imdb_movies.json", "w", encoding="utf-8") as f:
#         json.dump(movies_json_list, f, indent=4, ensure_ascii=False)


file = "imdb_movies.json"
with open(file, "r",encoding="utf-8") as f:
    old_movies_json_list = json.load(f)


not_found_titles = []
found = 0
not_found = 0

for movie in old_movies_json_list:
    if movie["Genre"] == "NOT_FOUND":
        not_found+=1
        not_found_titles.append(movie["Title"])
    else:
        found+=1

print(f'found: {found}')
print(f'not found {not_found}')
print(not_found_titles)