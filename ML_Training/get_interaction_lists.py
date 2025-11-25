from sklearn.model_selection import train_test_split

from get_user_item_dict import get_user_item_dict
import torch
import os

user_item_dict = get_user_item_dict()
def get_user_item_interaction_list():
    all_user_interactions = torch.tensor([])
    all_movie_interactions = torch.tensor([])

    for user, movies in user_item_dict.items():
        user_list = torch.tensor([user] * len(movies))

        movies = torch.tensor(movies)
        all_user_interactions = torch.concat([all_user_interactions,user_list],axis=0)
        all_movie_interactions = torch.concat([all_movie_interactions,movies],axis=0)


def get_user_item_list_test_train(train_proportion=0.8):

    folder = "train_test_split"
    os.makedirs(folder, exist_ok=True)

    split_file = os.path.join(folder, f"split_{train_proportion}.pt")

    # 1. Load if exists
    if os.path.exists(split_file):
        print("Loading existing split...")
        data = torch.load(split_file)
        return data["train_dict"], data["test_dict"], data["NUM_TRAIN_INTERACTIONS"]

    # create if it doesnt

    user_interactions_train = torch.tensor([])
    movie_interactions_train = torch.tensor([])

    user_interactions_test = torch.tensor([])
    movie_interactions_test = torch.tensor([])

    for user, movies in user_item_dict.items():
        user_list = torch.tensor([user] * len(movies))
        p = int(train_proportion * len(movies))

        perm = torch.randperm(len(movies))
        movies = movies[perm]

        movies = torch.tensor(movies)
        user_interactions_train = torch.cat((user_interactions_train,user_list[:p]))
        movie_interactions_train = torch.cat((movie_interactions_train,movies[:p]))

        user_interactions_test = torch.cat((user_interactions_test,user_list[p:]))
        movie_interactions_test = torch.cat((movie_interactions_test,movies[p:]))

    NUM_TRAIN_INTERACTIONS = len(movie_interactions_train)

    test_dict = {}
    for x in range(len(user_interactions_test)):
        user = int(user_interactions_test[x])
        movie = int(movie_interactions_test[x])

        if user not in test_dict:
            test_dict[user] = []

        test_dict[user].append(movie)


    train_dict = {}
    for x in range(len(user_interactions_train)):
        user = int(user_interactions_train[x])
        movie = int(movie_interactions_train[x])

        if user not in train_dict:
            train_dict[user] = []

        train_dict[user].append(movie)

    torch.save(
        {
            "train_dict": train_dict,
            "test_dict": test_dict,
            "NUM_TRAIN_INTERACTIONS": NUM_TRAIN_INTERACTIONS
        },
        split_file
    )


    return train_dict, test_dict, NUM_TRAIN_INTERACTIONS





