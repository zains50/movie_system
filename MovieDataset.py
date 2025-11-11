from torch.utils.data import Dataset
from get_interaction_lists import get_user_item_list_test_train
import random

train_dict, test_dict,NUM_TRAIN_INTERACTIONS = get_user_item_list_test_train(train_proportion=0.8)


class MovieDataset(Dataset):
    def __init__(self,n_user,n_item,user_features,movie_features):
        self.n_user = n_user
        self.n_item = n_item
        self.user_features = user_features
        self.movie_features = movie_features
        self.train_dict = train_dict
        self.NUM_TRAIN_INTERACTIONS = NUM_TRAIN_INTERACTIONS

        # flatten all interactions into a list of (user, pos_item) pairs
        self.interactions = []
        for user, movies in train_dict.items():
            for movie in movies:
                self.interactions.append((user, movie))

    def __len__(self):
        return self.NUM_TRAIN_INTERACTIONS

    def __getitem__(self, idx):
        # lets say idx is user id
        user,pos_item = self.interactions[idx]

        movies_not_watched = set(range(self.n_item-1)) - set(self.train_dict[user])
        movies_not_watched = list(movies_not_watched)
        random_neg_item = random.choice(movies_not_watched)



        return (
            self.user_features[user],
            self.movie_features[pos_item],
            self.movie_features[random_neg_item],
            user,
            pos_item,
            random_neg_item
        )

