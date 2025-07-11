#!/usr/bin/env python
# coding: utf-8

# # How to stream completions
#
# By default, when you request a completion from the OpenAI, the entire completion is generated before being sent back in a single response.
#
# If you're generating long completions, waiting for the response can take many seconds.
#
# To get responses sooner, you can 'stream' the completion as it's being generated. This allows you to start printing or processing the beginning of the completion before the full completion is finished.
#
# To stream completions, set `stream=True` when calling the chat completions or completions endpoints. This will return an object that streams back the response as [data-only server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format). Extract chunks from the `delta` field rather than the `message` field.
#
# ## Downsides
#
# Note that using `stream=True` in a production application makes it more difficult to moderate the content of the completions, as partial completions may be more difficult to evaluate. This may have implications for [approved usage](https://beta.openai.com/docs/usage-guidelines).
#
# ## Example code
#
# Below, this notebook shows:
# 1. What a typical chat completion response looks like
# 2. What a streaming chat completion response looks like
# 3. How much time is saved by streaming a chat completion
# 4. How to get token usage data for streamed chat completion response

# In[1]:


# !pip install openai


# In[2]:


# imports
import os
import time  # for measuring time duration of API calls

from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))


# ### 1. What a typical chat completion response looks like
#
# With a typical ChatCompletions API call, the response is first computed and then returned all at once.

# In[3]:


# Example of an OpenAI ChatCompletion request
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api

# record the time before the request is sent
start_time = time.time()

# send a ChatCompletion request to count to 100
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...",
        }
    ],
    temperature=0,
)
# calculate the time it took to receive the response
response_time = time.time() - start_time

# print the time delay and text received
print(f"Full response received {response_time:.2f} seconds after request")
print(f"Full response received:\n{response}")


# The reply can be extracted with `response.choices[0].message`.
#
# The content of the reply can be extracted with `response.choices[0].message.content`.

# In[4]:


reply = response.choices[0].message
print(f"Extracted reply: \n{reply}")

reply_content = response.choices[0].message.content
print(f"Extracted content: \n{reply_content}")


# ### 2. How to stream a chat completion
#
# With a streaming API call, the response is sent back incrementally in chunks via an [event stream](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format). In Python, you can iterate over these events with a `for` loop.
#
# Let's see what it looks like:

# In[5]:


# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/api-reference/streaming#chat/create-stream

# a ChatCompletion request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's 1+1? Answer in one word."}],
    temperature=0,
    stream=True,  # this time, we set stream=True
)

for chunk in response:
    print(chunk)
    print(chunk.choices[0].delta.content)
    print("****************")


# As you can see above, streaming responses have a `delta` field rather than a `message` field. `delta` can hold things like:
# - a role token (e.g., `{"role": "assistant"}`)
# - a content token (e.g., `{"content": "\n\n"}`)
# - nothing (e.g., `{}`), when the stream is over

# ### 3. How much time is saved by streaming a chat completion
#
# Now let's ask `gpt-4o-mini` to count to 100 again, and see how long it takes.

# In[6]:


# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/api-reference/streaming#chat/create-stream

# record the time before the request is sent
start_time = time.time()

# send a ChatCompletion request to count to 100
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...",
        }
    ],
    temperature=0,
    stream=True,  # again, we set stream=True
)
# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []
# iterate through the stream of events
for chunk in response:
    chunk_time = time.time() - start_time  # calculate the time delay of the chunk
    collected_chunks.append(chunk)  # save the event response
    chunk_message = chunk.choices[0].delta.content  # extract the message
    collected_messages.append(chunk_message)  # save the message
    print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

# print the time delay and text received
print(f"Full response received {chunk_time:.2f} seconds after request")
# clean None in collected_messages
collected_messages = [m for m in collected_messages if m is not None]
full_reply_content = "".join(collected_messages)
print(f"Full conversation received: {full_reply_content}")


# #### Time comparison
#
# In the example above, both requests took about 4 to 5 seconds to fully complete. Request times will vary depending on load and other stochastic factors.
#
# However, with the streaming request, we received the first token after 0.1 seconds, and subsequent tokens every ~0.01-0.02 seconds.

# ### 4. How to get token usage data for streamed chat completion response
#
# You can get token usage statistics for your streamed response by setting `stream_options={"include_usage": True}`. When you do so, an extra chunk will be streamed as the final chunk. You can access the usage data for the entire request via the `usage` field on this chunk. A few important notes when you set `stream_options={"include_usage": True}`:
# * The value for the `usage` field on all chunks except for the last one will be null.
# * The `usage` field on the last chunk contains token usage statistics for the entire request.
# * The `choices` field on the last chunk will always be an empty array `[]`.
#
# Let's see how it works using the example in 2.

# In[7]:


# Example of an OpenAI ChatCompletion request with stream=True and stream_options={"include_usage": True}

# a ChatCompletion request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's 1+1? Answer in one word."}],
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},  # retrieving token usage for stream response
)

for chunk in response:
    print(f"choices: {chunk.choices}\nusage: {chunk.usage}")
    print("****************")
