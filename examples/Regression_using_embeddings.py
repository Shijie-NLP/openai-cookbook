#!/usr/bin/env python
# coding: utf-8

# ## Regression using the embeddings
#
# Regression means predicting a number, rather than one of the categories. We will predict the score based on the embedding of the review's text. We split the dataset into a training and a testing set for all of the following tasks, so we can realistically evaluate performance on unseen data. The dataset is created in the [Get_embeddings_from_dataset Notebook](Get_embeddings_from_dataset.ipynb).
#
# We're predicting the score of the review, which is a number between 1 and 5 (1-star being negative and 5-star positive).

# In[2]:


from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.Score, test_size=0.2, random_state=42
)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)
preds = rfr.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"text-embedding-3-small performance on 1k Amazon reviews: mse={mse:.2f}, mae={mae:.2f}")


# In[3]:


bmse = mean_squared_error(y_test, np.repeat(y_test.mean(), len(y_test)))
bmae = mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))
print(f"Dummy mean prediction performance on Amazon reviews: mse={bmse:.2f}, mae={bmae:.2f}")


# We can see that the embeddings are able to predict the scores with an average error of 0.53 per score prediction. This is roughly equivalent to predicting half of reviews perfectly, and half off by one star.

# You could also train a classifier to predict the label, or use the embeddings within an existing ML model to encode free text features.
