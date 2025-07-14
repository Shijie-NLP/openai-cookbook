#!/usr/bin/env python
# coding: utf-8

# # How to use functions with a knowledge base
#
# This notebook builds on the concepts in the [argument generation](How_to_call_functions_with_chat_models.ipynb) notebook, by creating an agent with access to a knowledge base and two functions that it can call based on the user requirement.
#
# We'll create an agent that uses data from arXiv to answer questions about academic subjects. It has two functions at its disposal:
# - **get_articles**: A function that gets arXiv articles on a subject and summarizes them for the user with links.
# - **read_article_and_summarize**: This function takes one of the previously searched articles, reads it in its entirety and summarizes the core argument, evidence and conclusions.
#
# This will get you comfortable with a multi-function workflow that can choose from multiple services, and where some of the data from the first function is persisted to be used by the second.
#
# ## Walkthrough
#
# This cookbook takes you through the following workflow:
#
# - **Search utilities:** Creating the two functions that access arXiv for answers.
# - **Configure Agent:** Building up the Agent behaviour that will assess the need for a function and, if one is required, call that function and present results back to the agent.
# - **arXiv conversation:** Put all of this together in live conversation.
#

# In[2]:


import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from csv import writer

import arxiv
import pandas as pd
import tiktoken
from openai import OpenAI
from PyPDF2 import PdfReader
from scipy import spatial
from tenacity import retry, stop_after_attempt, wait_random_exponential
from termcolor import colored
from tqdm import tqdm


GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
client = OpenAI()


# ## Search utilities
#
# We'll first set up some utilities that will underpin our two functions.
#
# Downloaded papers will be stored in a directory (we use ```./data/papers``` here). We create a file ```arxiv_library.csv``` to store the embeddings and details for downloaded papers to retrieve against using ```summarize_text```.

# In[3]:


directory = "./data/papers"

# Check if the directory already exists
if not os.path.exists(directory):
    # If the directory doesn't exist, create it and any necessary intermediate directories
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully.")
else:
    # If the directory already exists, print a message indicating it
    print(f"Directory '{directory}' already exists.")


# In[4]:


# Set a directory to store downloaded papers
data_dir = os.path.join(os.curdir, "data", "papers")
paper_dir_filepath = "./data/papers/arxiv_library.csv"

# Generate a blank dataframe where we can store downloaded files
# df = pd.DataFrame([])
# df.to_csv(paper_dir_filepath)


# In[5]:


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def get_articles(query, library=paper_dir_filepath, top_k=10):
    """This function gets the top_k articles based on a user's query, sorted by relevance.
    It also downloads the files and stores them in arxiv_library.csv to be retrieved by the read_article_and_summarize.
    """
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=top_k)
    result_list = []
    for result in client.results(search):
        result_dict = {}
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})

        # Taking the first url provided
        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_list.append(result_dict)

        # Store references in library file
        response = embedding_request(text=result.title)
        file_reference = [
            result.title,
            result.download_pdf(data_dir),
            response.data[0].embedding,
        ]

        # Write to file
        with open(library, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(file_reference)
            f_object.close()
    return result_list


# In[6]:


# Test that the search is working
# result_output = get_articles("ppo reinforcement learning")
# print(result_output[0])


# In[7]:


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> list[str]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = embedding_request(query)
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["filepath"], relatedness_fn(query_embedding, row["embedding"])) for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n]


# In[8]:


def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a PDF reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    """Returns successive n-sized chunks from provided text."""
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of the sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def extract_chunk(content, template_prompt):
    """This function applies a prompt to some input content. In this case it returns a summarised chunk of text"""
    prompt = template_prompt + content
    response = client.chat.completions.create(
        model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response.choices[0].message.content


def summarize_text(query):
    """This function does the following:
    - Reads in the arxiv_library.csv file in including the embeddings
    - Finds the closest file to the user's query
    - Scrapes the text out of the file and chunks it
    - Summarizes each chunk in parallel
    - Does one final summary and returns this to the user"""

    # A prompt to dictate how the recursive summarisations should approach the input paper
    summary_prompt = (
        """Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"""
    )

    # If the library is empty (no searches have been performed yet), we perform one and download the results
    library_df = pd.read_csv(paper_dir_filepath).reset_index()
    if len(library_df) == 0:
        print("No papers searched yet, downloading first.")
        get_articles(query)
        print("Papers downloaded, continuing")
        library_df = pd.read_csv(paper_dir_filepath).reset_index()
    else:
        print("Existing papers found... Articles:", len(library_df))
    library_df.columns = ["title", "filepath", "embedding"]
    library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)
    strings = strings_ranked_by_relatedness(query, library_df, top_n=1)
    print("Chunking text from paper")
    pdf_text = read_pdf(strings[0])

    # Initialise tokeniser
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results = ""

    # Chunk up the document into 1500 token chunks
    chunks = create_chunks(pdf_text, 1500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    print("Summarizing each chunk of text")

    # Parallel process the summaries
    with ThreadPoolExecutor(max_workers=len(text_chunks)) as executor:
        futures = [executor.submit(extract_chunk, chunk, summary_prompt) for chunk in text_chunks]
        with tqdm(total=len(text_chunks)) as pbar:
            for _ in as_completed(futures):
                pbar.update(1)
        for future in futures:
            data = future.result()
            results += data

    # Final summary
    print("Summarizing into overall summary")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Write a summary collated from this collection of key points extracted from an academic paper.
                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
                        User query: {query}
                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                        Key points:\n{results}\nSummary:\n""",
            }
        ],
        temperature=0,
    )
    return response


# In[9]:


# Test the summarize_text function works
chat_test_response = summarize_text("PPO reinforcement learning sequence generation")


# In[10]:


print(chat_test_response.choices[0].message.content)


# ## Configure Agent
#
# We'll create our agent in this step, including a ```Conversation``` class to support multiple turns with the API, and some Python functions to enable interaction between the ```ChatCompletion``` API and our knowledge base functions.

# In[11]:


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# In[12]:


class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )


# In[13]:


# Initiate our get_articles and read_article_and_summarize functions
arxiv_functions = [
    {
        "name": "get_articles",
        "description": """Use this function to get academic papers from arXiv to answer user questions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": """
                            User query in JSON. Responses should be summarized and should include the article URL reference
                            """,
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_article_and_summarize",
        "description": """Use this function to read whole papers and provide a summary for users.
        You should NEVER call this function before get_articles has been called in the conversation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": """
                            Description of the article in plain text based on the user's query
                            """,
                }
            },
            "required": ["query"],
        },
    },
]


# In[14]:


def chat_completion_with_function_execution(messages, functions=None):
    """This function makes a ChatCompletion API call with the option of adding functions"""
    response = chat_completion_request(messages, functions)
    full_message = response.choices[0]
    if full_message.finish_reason == "function_call":
        print("Function generation requested, calling function")
        return call_arxiv_function(messages, full_message)
    else:
        print("Function not required, responding to user")
        return response


def call_arxiv_function(messages, full_message):
    """Function calling function which executes function calls when the model believes it is necessary.
    Currently extended by adding clauses to this if statement."""

    if full_message.message.function_call.name == "get_articles":
        try:
            parsed_output = json.loads(full_message.message.function_call.arguments)
            print("Getting search results")
            results = get_articles(parsed_output["query"])
        except Exception as e:
            print(parsed_output)
            print("Function execution failed")
            print(f"Error message: {e}")
        messages.append(
            {
                "role": "function",
                "name": full_message.message.function_call.name,
                "content": str(results),
            }
        )
        try:
            print("Got search results, summarizing content")
            response = chat_completion_request(messages)
            return response
        except Exception as e:
            print(type(e))
            raise Exception("Function chat request failed")

    elif full_message.message.function_call.name == "read_article_and_summarize":
        parsed_output = json.loads(full_message.message.function_call.arguments)
        print("Finding and reading paper")
        summary = summarize_text(parsed_output["query"])
        return summary

    else:
        raise Exception("Function does not exist and cannot be called")


# ## arXiv conversation
#
# Let's put this all together by testing our functions out in conversation.

# In[15]:


# Start with a system message
paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions.
You summarize the papers clearly so the customer can decide which to read to answer their question.
You always provide the article_url and title so the user can understand the name of the paper and click through to access it.
Begin!"""
paper_conversation = Conversation()
paper_conversation.add_message("system", paper_system_message)


# In[16]:


# Add a user message
paper_conversation.add_message("user", "Hi, how does PPO reinforcement learning work?")
chat_response = chat_completion_with_function_execution(
    paper_conversation.conversation_history, functions=arxiv_functions
)
assistant_message = chat_response.choices[0].message.content
paper_conversation.add_message("assistant", assistant_message)
print(assistant_message)


# In[17]:


# Add another user message to induce our system to use the second tool
paper_conversation.add_message(
    "user",
    "Can you read the PPO sequence generation paper for me and give me a summary",
)
updated_response = chat_completion_with_function_execution(
    paper_conversation.conversation_history, functions=arxiv_functions
)
print(updated_response.choices[0].message.content)
