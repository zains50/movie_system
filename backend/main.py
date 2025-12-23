from typing import Union 
from fastapi import FastAPI
from pydantic import BaseModel

from backend.process_search_query import search_router  # import the router

app = FastAPI()

app.include_router(search_router)
items_db = {}


class Item(BaseModel): 
    name: str
    price: float
    if_offer: Union[bool, None] = None

class User(BaseModel):
    name: str
    password: str
    email: str

# create an item
@app.post("/users/{user_id}")
def create_user(item_id: int, user: User):
    # items_db[item_id] = item
    # print(f"New items: {items_db}")
    return {"message" : "Item stored", "item":0}

@app.get("/")
def read_root():
    return {"Hello" : "i am a noobyyy"}

# create an item
@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item):
    items_db[item_id] = item
    print(f"New items: {items_db}")
    return {"message" : "Item stored", "item":item}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items_db:
        return {"error": "Item not found"}

    return items_db[item_id]

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id":item_id}

