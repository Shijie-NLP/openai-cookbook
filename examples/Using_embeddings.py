#!/usr/bin/env python
# coding: utf-8

# ## Using embeddings
#
# This notebook contains some helpful snippets you can use to embed text with the `text-embedding-3-small` model via the OpenAI API.

# In[ ]:


from openai import OpenAI


client = OpenAI()

embedding = client.embeddings.create(input="Your text goes here", model="text-embedding-3-small").data[0].embedding
print(len(embedding))


# It's recommended to use the 'tenacity' package or another exponential backoff implementation to better manage API rate limits, as hitting the API too much too fast can trigger rate limits. Using the following function ensures you get your embeddings as fast as possible.

# In[ ]:


# Negative example (slow and rate-limited)
from openai import OpenAI


client = OpenAI()

num_embeddings = 10000  # Some large number
for i in range(num_embeddings):
    embedding = client.embeddings.create(input="Your text goes here", model="text-embedding-3-small").data[0].embedding
    print(len(embedding))


# In[2]:


# Best practice
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


client = OpenAI()


# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    return client.embeddings.create(input=[text], model=model).data[0].embedding


embedding = get_embedding("Your text goes here", model="text-embedding-3-small")
print(len(embedding))
