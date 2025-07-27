#!/usr/bin/env python
# coding: utf-8

# ## Zero-shot classification with embeddings
#
# In this notebook we will classify the sentiment of reviews using embeddings and zero labeled data! The dataset is created in the [Get_embeddings_from_dataset Notebook](Get_embeddings_from_dataset.ipynb).
#
# We'll define positive sentiment to be 4- and 5-star reviews, and negative sentiment to be 1- and 2-star reviews. 3-star reviews are considered neutral and we won't use them for this example.
#
# We will perform zero-shot classification by embedding descriptions of each class and then comparing new samples to those class embeddings.

# In[1]:


from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


EMBEDDING_MODEL = "text-embedding-3-small"

datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# convert 5-star rating to binary sentiment
df = df[df.Score != 3]
df["sentiment"] = df.Score.replace({1: "negative", 2: "negative", 4: "positive", 5: "positive"})


# ### Zero-Shot Classification
# To perform zero shot classification, we want to predict labels for our samples without any training. To do this, we can simply embed short descriptions of each label, such as positive and negative, and then compare the cosine distance between embeddings of samples and label descriptions.
#
# The highest similarity label to the sample input is the predicted label. We can also define a prediction score to be the difference between the cosine distance to the positive and to the negative label. This score can be used for plotting a precision-recall curve, which can be used to select a different tradeoff between precision and recall, by selecting a different threshold.

# In[2]:


from sklearn.metrics import PrecisionRecallDisplay
from utils.embeddings_utils import cosine_similarity, get_embedding


def evaluate_embeddings_approach(labels=["negative", "positive"], model=EMBEDDING_MODEL):
    label_embeddings = [get_embedding(label, model=model) for label in labels]

    def label_score(review_embedding, label_embeddings):
        return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(
            review_embedding, label_embeddings[0]
        )

    probas = df["embedding"].apply(lambda x: label_score(x, label_embeddings))
    preds = probas.apply(lambda x: "positive" if x > 0 else "negative")

    report = classification_report(df.sentiment, preds)
    print(report)

    display = PrecisionRecallDisplay.from_predictions(df.sentiment, probas, pos_label="positive")
    _ = display.ax_.set_title("2-class Precision-Recall curve")


evaluate_embeddings_approach(labels=["negative", "positive"], model=EMBEDDING_MODEL)


# We can see that this classifier already performs extremely well. We used similarity embeddings and the simplest possible label name. Let's try to improve on this by using more descriptive label names, and search embeddings.

# In[4]:


evaluate_embeddings_approach(
    labels=["An Amazon review with a negative sentiment.", "An Amazon review with a positive sentiment."]
)


# Using the search embeddings and descriptive names leads to an additional improvement in performance.

# In[5]:


evaluate_embeddings_approach(
    labels=["An Amazon review with a negative sentiment.", "An Amazon review with a positive sentiment."]
)


# As shown above, zero-shot classification with embeddings can lead to great results, especially when the labels are more descriptive than just simple words.
