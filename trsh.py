import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import mfnn

df_ratings_in = pd.read_csv('data/ratings.csv')

item_mapping = pd.read_csv('data/movies.csv')
item_mapping = item_mapping[['movieId', 'title']]
item_mapping = dict(item_mapping.values)

rec = mfnn.MFNNRecommender(
    data=df_ratings_in,
    user_field='userId',
    item_field='movieId',
    rating_field='rating',
)

rec.fit(dim=20, epochs=200)
print(rec.recommend(1, 10))
rec.save('200epochs')

rec_new = mfnn.MFNNRecommender(onload=True)
rec_new.load('200epochs')

print(rec_new.recommend(1, 10, item_mapping))