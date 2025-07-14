#!/usr/bin/env python
# coding: utf-8

# # Translate a book written in LaTeX from Slovenian into English
#
# With permission of the author, we will demonstrate how to translate the book [Euclidean Plane Geometry](https://sites.google.com/site/projektivna/), written by Milan Mitrović from Slovenian into English, without modifying any of the LaTeX commands.
#
# To achieve this, we will first split the book into chunks, each roughly a page long, then translate each chunk into English, and finally stitch them back together.

# ## 1. Read in the data

# In[2]:


import tiktoken
from openai import OpenAI


client = OpenAI()

# OpenAI tiktoken tokenizer: https://github.com/openai/tiktoken
# we use it to count the number of tokens in the text
tokenizer = tiktoken.get_encoding("o200k_base")

with open("data/geometry_slovenian.tex", "r") as f:
    text = f.read()


# ### 1.1 Count the tokens in each chunk

# In[3]:


chunks = text.split("\n\n")
ntokens = []
for chunk in chunks:
    ntokens.append(len(tokenizer.encode(chunk)))
print("Size of the largest chunk: ", max(ntokens))
print("Number of chunks: ", len(chunks))


# It turns out that a double newline is a good separator in this case, in order not to break the flow of the text. Also no individual chunk is larger than 1211 tokens. The model we will use is gpt-4o, which has a limit of 16,384 tokens, so we don't need to worry about breaking the chunks down further.
#
# We will group the shorter chunks into chunks of around 15000 tokens, to increase the coherence of the text, and decrease the frequency of breaks within the text.

# In[4]:


def group_chunks(chunks, ntokens, max_len=15000, hard_max_len=16000):
    """
    Group very short chunks, to form approximately page long chunks.
    """
    batches = []
    cur_batch = ""
    cur_tokens = 0

    # iterate over chunks, and group the short ones together
    for chunk, ntoken in zip(chunks, ntokens):
        # discard chunks that exceed hard max length
        if ntoken > hard_max_len:
            print(
                f"Warning: Chunk discarded for being too long ({ntoken} tokens > {hard_max_len} token limit). Preview: '{chunk[:50]}...'"
            )
            continue

        # if room in current batch, add new chunk
        if cur_tokens + 1 + ntoken <= max_len:
            cur_batch += "\n\n" + chunk
            cur_tokens += 1 + ntoken  # adds 1 token for the two newlines
        # otherwise, record the batch and start a new one
        else:
            batches.append(cur_batch)
            cur_batch = chunk
            cur_tokens = ntoken

    if cur_batch:  # add the last batch if it's not empty
        batches.append(cur_batch)

    return batches


chunks = group_chunks(chunks, ntokens)
print(len(chunks))


# Notice that adding a sample untranslated and translated first command, where only the content of the chapter name needs to be translated, helps to get more consistent results.
#
# The format of the prompt sent to the model consists of:
# 1. A high level instruction to translate only the text, but not commands into the desired language
# 2. A sample untranslated command, where only the content of the chapter name needs to be translated
# 3. The chunk of text to be translated
# 4. The translated sample command from 2, which shows the model the beginning of the translation process
#
# The expected output is the translated chunk of text.

# In[5]:


def translate_chunk(
    chunk,
    model="gpt-4o",
    dest_language="English",
    sample_translation=(
        r"\poglavje{Osnove Geometrije} \label{osn9Geom}",
        r"\chapter{The basics of Geometry} \label{osn9Geom}",
    ),
):
    prompt = f'''Translate only the text from the following LaTeX document into {dest_language}. Leave all LaTeX commands unchanged

"""
{sample_translation[0]}
{chunk}"""

{sample_translation[1]}
'''
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=15000,
    )
    result = response.choices[0].message.content.strip()
    result = result.replace('"""', "")  # remove the double quotes, as we used them to surround the text
    return result


print(translate_chunk(chunks[2], model="gpt-4o", dest_language="English"))


# We can see here that this one chunk in particular translates only the text, but leaves LaTeX commands intact.
#
# Let's now translate all the chunks in the book - this will take 2-3 hours, as we're processing requests sequentially.

# In[ ]:


dest_language = "English"

translated_chunks = []
for i, chunk in enumerate(chunks):
    print(str(i + 1) + " / " + str(len(chunks)))
    # translate each chunk
    translated_chunks.append(translate_chunk(chunk, model="gpt-4o", dest_language=dest_language))

# join the chunks together
result = "\n\n".join(translated_chunks)

# save the final result
with open(f"data/geometry_{dest_language}.tex", "w") as f:
    f.write(result)


# In[ ]:


from concurrent.futures import ThreadPoolExecutor, as_completed


# Function to translate a single chunk
def translate_chunk_wrapper(chunk, model="gpt-4o", dest_language="English"):
    return translate_chunk(chunk, model=model, dest_language=dest_language)


# Set the destination language
dest_language = "English"

# List to store translated chunks
translated_chunks = []

# Use ThreadPoolExecutor to parallelize the translation
with ThreadPoolExecutor(max_workers=5) as executor:
    # Submit all translation tasks
    futures = {
        executor.submit(translate_chunk_wrapper, chunk, "gpt-4o", dest_language): i for i, chunk in enumerate(chunks)
    }

    # Process completed tasks as they finish
    for future in as_completed(futures):
        i = futures[future]
        try:
            translated_chunk = future.result()
            translated_chunks.append(translated_chunk)
            print(f"Chunk {i + 1} / {len(chunks)} translated.")
        except Exception as e:
            print(f"Chunk {i + 1} failed with exception: {e}")

# Join the translated chunks together
result = "\n\n".join(translated_chunks)

# Save the final result
with open(f"data/geometry_{dest_language}.tex", "w") as f:
    f.write(result)
