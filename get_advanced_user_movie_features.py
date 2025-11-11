from data_extraction import  get_all_features_numpy
import json
import requests

movies_arr, ratings_arr, users_arr = get_all_features_numpy()

def get_omdb_movie_dict_list():
    print(movies_arr)
    print(movies_arr.shape)
    file = "json_file/movies_omdb_1_1.json"
    omdb_movie_dict_list = []

    # Open and load the JSON
    with open(file, "r",encoding="utf-8") as f:
        data = json.load(f)
    i=0
    for x in movies_arr:
        if x[1] == "FAKE MOVIE":
            omdb_movie_dict_list.append("")
        else:
            omdb_movie_dict_list.append(data[i])
            i+=1

    return omdb_movie_dict_list

def get_movie_title_and_summary_list():
    omdb_movie_dict_list = get_omdb_movie_dict_list()
    movie_title_summ_list = []
    for x in omdb_movie_dict_list:
        if x == "":
            movie_title_summ_list.append("")
        elif x["Genre"] == "NOT_FOUND":
            movie_title_summ_list.append(f"{x['Title']}: Not found")
        else:
            movie_title_summ_list.append(f"{x['Title']}: {x['Plot']}")
    return movie_title_summ_list

def get_movie_poster_link_list():
    omdb_movie_dict_list = get_omdb_movie_dict_list()
    poster_link_list = []
    for x in omdb_movie_dict_list:
        if x == "":
            poster_link_list.append("")
        elif x["Genre"] == "NOT_FOUND":
            poster_link_list.append(f"NOT_FOUND")
        else:
                poster_link_list.append(f"{x['Poster']}")
    return poster_link_list

def save_posters():
    omdb_movie_dict_list = get_omdb_movie_dict_list()
    poster_link_list = get_movie_poster_link_list()
    for i,p in enumerate(poster_link_list):
        if p is not "" and p is not "NOT_FOUND":
            url = p
            file_name = f'movie_posters/{i}_{omdb_movie_dict_list[i]["Title"]}.jpg'
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(file_name, "wb") as f:
                        print(f'downloaded movie: {i}_{omdb_movie_dict_list[i]["Title"]}')
                        f.write(response.content)
                else:
                    print(f"Failed to download: {url}, status code {response.status_code}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")



sum = get_movie_title_and_summary_list()
print(sum[0])