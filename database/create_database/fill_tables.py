import json 
import psycopg2
from tqdm import tqdm
from psycopg2 import sql

from datetime import datetime


## DATA BASE CONNECTION

with open ("data/db_config.json") as f:
    db_config = json.load(f)

conn = psycopg2.connect(**db_config)
conn.autocommit = True 
cursor = conn.cursor()


## CODE TO FILL TABLES

with open('data/movie_info_with_uuid.json', encoding="utf-8") as f:
    genres = []
    movie_information = json.load(f)

## DONE THIS ONE 
def fill_genre_table(movie_information_json):
    
    # Step 1) Get a list of genres that our movies use 
    all_genres = []
    for m in tqdm(movie_information_json):
        genres = m["Genre"].split(", ")
        for g in genres:
            all_genres.append(g)
    
    # Get unique values
    all_genres = list(set(all_genres))

    query = sql.SQL("INSERT INTO GENRE ({col1}, {col2}) VALUES (%s, %s)").format( 
        col1=sql.Identifier("genre_id"),
        col2=sql.Identifier("genre_name"),
    )


    for i,g in enumerate(all_genres):
        cursor.execute(query, (i,g))

## DONE THIS ONE 
def fill_person_table(movie_information_json):
    # same logic as genre but with movies 
    all_people = []
    for m in tqdm(movie_information_json):
        directors = m.get("Director","N/A").split(", ")
        writers = m.get("Writer","N/A").split(", ")
        actors = m.get("Actors","N/A").split(", ")

        # some names have (stuff) so get rid of anything with brackets
        for d in directors:
            all_people.append(d.split(" (")[0])
        for w in writers:
            all_people.append(w.split(" (")[0])
        for a in actors:
            all_people.append(a.split(" (")[0])
    
    all_people = list(set(all_people))

    query = sql.SQL("INSERT INTO PERSON ({col1}, {col2}) VALUES (%s, %s)").format( 
        col1=sql.Identifier("person_id"),
        col2=sql.Identifier("person_name"),
    )


    for i,g in enumerate(all_people):
        cursor.execute(query, (i,g))

def fill_roles_table():
    # easy
    roles = ["Writer","Actor","Director"]
    
    query = sql.SQL("INSERT INTO ROLES ({col1}, {col2}) VALUES (%s, %s)").format( 
        col1=sql.Identifier("role_id"),
        col2=sql.Identifier("role_name"),
    )


    for i,r in enumerate(roles):
        cursor.execute(query, (i,r))

def fill_content_rating_table(movie_information_json):
    content_ratings = []
    for CR in tqdm(movie_information_json):
        ratings = CR.get("Rated","N/A").split(", ")

        # some names have (stuff) so get rid of anything with brackets
        for r in ratings:
            content_ratings.append(r)
    
    content_ratings = list(set(content_ratings))
    desc = "WILL ADD SOON"
    
    query = sql.SQL("INSERT INTO CONTENT_RATING ({col1}, {col2}, {col3}) VALUES (%s, %s, %s)").format( 
        col1=sql.Identifier("content_rating_id"),
        col2=sql.Identifier("content_rating"),
        col3=sql.Identifier("content_rating_description"),
    )

    for i,r in enumerate(content_ratings):
        cursor.execute(query, (i,r, desc))

def fill_movie_table(movie_information_json):
    movies = []
    cursor.execute("DELETE FROM MOVIE")

    query = sql.SQL("INSERT INTO MOVIE ({col1}, {col2}, {col3}, {col4}, {col5}, {col6}, {col7}, {col8}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)").format( 
            col1=sql.Identifier("movie_id"),
            col2=sql.Identifier("movie_title"),
            col3=sql.Identifier("content_rating_id"),
            col4=sql.Identifier("plot"),
            col5=sql.Identifier("year"),
            col6=sql.Identifier("released"),
            col7=sql.Identifier("runtime_mins"),
            col8=sql.Identifier("poster_link"),
            
    )
    
    for movie in tqdm(movie_information_json):
        if movie["Genre"] == "NOT_FOUND":
            continue 

        added_movie_ids = []

        movie_title = movie["Title"]
        movie_rating = movie["Rated"]
        movie_plot = movie["Plot"]
        movie_year = movie["Year"].split("–")[0]
        movie_release_date = movie["Released"]
        movie_runtime = movie["Runtime"].split(" ")[0]
        movie_poster = movie["Poster"]
        movie_id = movie["UUID"]

        if movie_id in added_movie_ids:
            continue;

        added_movie_ids.append(movie_id)
    

        # First get rating id 
        cursor.execute("SELECT * FROM CONTENT_RATING where content_rating = %s", (movie_rating,))
        rows = cursor.fetchall()
        movie_rating_id,row_name,_ = rows[0]
    
        # print(movie_release_date)
        try:
            movie_release_date = datetime.strptime(movie_release_date, "%d %b %Y").date()
        except:
            movie_release_date = None
        
        if movie_runtime == "N/A":
            movie_runtime = None 

        # print(movie_release_date)
        cursor.execute(query, (movie_id, movie_title,movie_rating_id,movie_plot, movie_year, movie_release_date, movie_runtime, movie_poster))
       

def fill_movie_genre_table(movie_information_json):
    cursor.execute("DELETE FROM movie_genre")
    movie_ids = []
    # needs movie_id, genre_id
    for movie in tqdm(movie_information_json):
        if movie["Genre"] == "NOT_FOUND":
            continue 
        movie_year = movie["Year"][:4]
        movie_title = movie["Title"]
        genres = movie["Genre"].split(", ")

        cursor.execute("SELECT movie_id from MOVIE where movie_title = %s AND year =  %s" , (movie_title,movie_year))
        movie_id = cursor.fetchall()[0]

        if movie_id in movie_ids:
            continue # means  its a duplicate

        for g in genres:
            # stuff 
            cursor.execute("SELECT genre_id FROM GENRE where genre_name = %s" , (g,))
            genre_id = cursor.fetchall()[0]



            # print(movie_title, g)
            # print(movie_id, genre_id)
            query = sql.SQL("INSERT INTO movie_genre ({col1}, {col2}) VALUES (%s, %s)").format( 
                col1=sql.Identifier("movie_id"),
                col2=sql.Identifier("genre_id"),
            )

            cursor.execute(query, (movie_id, genre_id)) 
            movie_ids.append(movie_id)

# cursor.execute("DELETE FROM movie_genre")
fill_movie_table(movie_information)
fill_movie_genre_table(movie_information)
