## CSCI4050U Final Project - Movie Recommendation System

### Problem Statement

- We want to make a movie recommendation system. The recommendation system should have the following properties:
- Be able to utilize user information (age, gender, occupation)
- Be able to utilize a list of movies a user has watched.
- Be able to predict movies a user will watch based on past interaction information that the model will learn on
- Be able to create recommendations for new users with new interactions
- Be able to recommend new movies that have not been seen during training

### How we are going to achieve our goals

- We want to use the user-item interaction matrix to capture user preferences. We want our model to encode information about a user such as personal identification information and movies a user has watched. We will then use this encoding to generate a score = (u,m) between a user and the encoding of every movie.

- We want the model to learn a movie encoder where films with similar genres or themes lie close together, and dissimilar films are positioned far apart. At the same time, the model should learn a user encoder that positions users near the movie vectors that they will be interested in.

- We use the user–item interaction matrix to capture taste patterns. For example the model may learn “Users that watch comedy movies would be more interested in movies that are light-hearted compared  to horror movies”. To help our model learn this, we incorporate movie poster features and plot text features to enrich each movie’s representation with visual and textual information.

#### To view our neural network please check
```ML_Training/Neural_Network.py```

#### To view training procedure please check
```ML_Training/main.py```

#### To view our model deployment please run
```Model_Deployment/main.py```
### To run our model you can do the following.
- Go to ML_Training/main.py
- You can run with the default parameters or pick your own by editing
```    
train(embed_size=256,num_layers=2,batch_size=1048,epochs=100,lr=1e-4,weight_decay=5e-4,gpu=0,save_every=2,use_text_data=True, use_poster_data=True)
```
## Dependencies

### Core Libraries
- Python 3.9.21
- torch 2.5.1
- torch_scatter 2.1.2
- numpy
- pandas
- tqdm
- sentence_transformers

### Model Deployment Dependencies
- customtkinter: 5.2.2
- pillow