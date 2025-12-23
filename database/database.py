import json 
import psycopg2
from tqdm import tqdm
from psycopg2 import sql
from datetime import datetime
import os
from dotenv import load_dotenv
class database:
    def __init__(self):
        print("DATABASE.py : CONNECTING TO POSTGRES DATABASE")
        load_dotenv()   

        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
        )
        conn.autocommit = True 
        self.cursor = conn.cursor()
        print("DATABASE.py : CONNECTED TO POSTGRES DATABASE")

    def get_movie_from_uuid(self, uuid):
        #  SELECT movie_title FROM MOVIE where movie_id='af62396b-5c8b-4f05-b3db-418f923046b1';
        query = sql.SQL("SELECT movie_title FROM MOVIE where movie_id=%s")
        self.cursor.execute(query, (uuid, ))
        return self.cursor.fetchall()


DATABASE_CONNECTION = database()
