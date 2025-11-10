import json
import os
from data_extraction import get_movies_numpy
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv('API_KEY')

def fix_title(title):
    articles = {'The', 'A', 'An'}
    # Strip any extra whitespace first
    title = title.strip()
    if ',' in title:
        parts = title.rsplit(',', 1)  # Split on the last comma only
        if len(parts) == 2:
            article = parts[1].strip()  # Remove extra spaces
            if article in articles:
                return f"{article} {parts[0].strip()}"
    return title



file = "json_file/movies_omdb_1_1.json"
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

# new_movies_json_list = []

# movies = get_movies_numpy()

# for i,movie in enumerate(old_movies_json_list):
#     if movie["Genre"] == "NOT_FOUND":
#         fixed_title = fix_title(movie["Title"])
#         actual_title = movies[i,1]
#         movie_year  = actual_title[1].split("(")[-1][:-1]
#
#         url = f"http://www.omdbapi.com/?t={fixed_title}&y={movie_year}&apikey={API_KEY}"
#         try:
#             response = requests.get(url, timeout=10)
#             data = response.json()
#             print(data)
#             if data.get("Response") == "True":
#                 new_movies_json_list.append(data)
#
#             elif data.get("Response") == "False":
#                 json_obj = {'Title':fixed_title, 'Genre':"NOT_FOUND"}
#                 new_movies_json_list.append(json_obj)
#         except Exception:
#             print('api had an error')
#
#     else:
#         new_movies_json_list.append(old_movies_json_list[i])
#
#
#     with open("json_file/movies_omdb_1_1.json", "w", encoding="utf-8") as f:
#         json.dump(new_movies_json_list, f, indent=4, ensure_ascii=False)