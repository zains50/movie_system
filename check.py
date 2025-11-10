from data_extraction import get_movies_numpy
import json
movies = get_movies_numpy()
with open("json_file/movies_omdb_1_1.json", "r",encoding="utf-8") as f:
    old_movies_json_list = json.load(f)

i = 0
for x in old_movies_json_list:
    print(f'{movies[i][1]},{x["Title"]}')
    i+=1