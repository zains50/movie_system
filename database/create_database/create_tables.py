import psycopg2
import json 
from psycopg2 import sql 

# Code to create tables 
def add_tables_to_database(): 
    with open ("data/db_config.json") as f:
        db_config = json.load(f)

    conn = psycopg2.connect(**db_config)
    conn.autocommit = True 
    cursor = conn.cursor()

    with open("sql_scripts/create_tables.sql") as f:
        cursor.execute(f.read())

    print("Tables created")

    cursor.close()
    conn.close() 

add_tables_to_database()