#!/usr/bin/env python
# coding: utf-8

# # Azure chat completions example
#
# This example will cover chat completions using the Azure OpenAI service. It also includes information on content filtering.

# ## Setup
#
# First, we install the necessary dependencies and import the libraries we will be using.

# In[ ]:


import os

import dotenv
import openai


dotenv.load_dotenv()


# ### Authentication
#
# The Azure OpenAI service supports multiple authentication mechanisms that include API keys and Azure Active Directory token credentials.

# In[2]:


use_azure_active_directory = False  # Set this flag to True if you are using Azure Active Directory


# #### Authentication using API key
#
# To set up the OpenAI SDK to use an *Azure API Key*, we need to set `api_key` to a key associated with your endpoint (you can find this key in *"Keys and Endpoints"* under *"Resource Management"* in the [Azure Portal](https://portal.azure.com)). You'll also find the endpoint for your resource here.

# In[13]:


if not use_azure_active_directory:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    client = openai.AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2023-09-01-preview")


# #### Authentication using Azure Active Directory
# Let's now see how we can autheticate via Azure Active Directory. We'll start by installing the `azure-identity` library. This library will provide the token credentials we need to authenticate and help us build a token credential provider through the `get_bearer_token_provider` helper function. It's recommended to use `get_bearer_token_provider` over providing a static token to `AzureOpenAI` because this API will automatically cache and refresh tokens for you.
#
# For more information on how to set up Azure Active Directory authentication with Azure OpenAI, see the [documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity).

# In[5]:


from azure.identity import DefaultAzureCredential, get_bearer_token_provider


if use_azure_active_directory:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        ),
        api_version="2023-09-01-preview",
    )


# > Note: the AzureOpenAI infers the following arguments from their corresponding environment variables if they are not provided:
#
# - `api_key` from `AZURE_OPENAI_API_KEY`
# - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
# - `api_version` from `OPENAI_API_VERSION`
# - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
#

# ## Deployments
#
# In this section we are going to create a deployment of a GPT model that we can use to create chat completions.

# ### Deployments: Create in the Azure OpenAI Studio
# Let's deploy a model to use with chat completions. Go to https://portal.azure.com, find your Azure OpenAI resource, and then navigate to the Azure OpenAI Studio. Click on the "Deployments" tab and then create a deployment for the model you want to use for chat completions. The deployment name that you give the model will be used in the code below.

# In[4]:


deployment = "gpt-4o-mini"  # Fill in the deployment name from the portal here


# ## Create chat completions
#
# Now let's create a chat completion using the client we built.

# In[ ]:


# For all possible arguments see https://platform.openai.com/docs/api-reference/chat-completions/create
response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
    temperature=0,
)

print(f"{response.choices[0].message.role}: {response.choices[0].message.content}")


# ### Create a streaming chat completion
#
# We can also stream the response.

# In[ ]:


response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
    temperature=0,
    stream=True,
)

for chunk in response:
    if len(chunk.choices) > 0:
        delta = chunk.choices[0].delta

        if delta.role:
            print(delta.role + ": ", end="", flush=True)
        if delta.content:
            print(delta.content, end="", flush=True)


# ### Content filtering
#
# Azure OpenAI service includes content filtering of prompts and completion responses. You can learn more about content filtering and how to configure it [here](https://learn.microsoft.com/azure/ai-services/openai/concepts/content-filter).
#
# If the prompt is flagged by the content filter, the library will raise a `BadRequestError` exception with a `content_filter` error code. Otherwise, you can access the `prompt_filter_results` and `content_filter_results` on the response to see the results of the content filtering and what categories were flagged.

# #### Prompt flagged by content filter

# In[ ]:


import json


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "<text violating the content policy>"},
]

try:
    completion = client.chat.completions.create(
        messages=messages,
        model=deployment,
    )
except openai.BadRequestError as e:
    err = json.loads(e.response.text)
    if err["error"]["code"] == "content_filter":
        print("Content filter triggered!")
        content_filter_result = err["error"]["innererror"]["content_filter_result"]
        for category, details in content_filter_result.items():
            print(f"{category}:\n filtered={details['filtered']}\n severity={details['severity']}")


# ### Checking the result of the content filter

# In[ ]:


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the biggest city in Washington?"},
]

completion = client.chat.completions.create(
    messages=messages,
    model=deployment,
)
print(f"Answer: {completion.choices[0].message.content}")

# prompt content filter result in "model_extra" for azure
prompt_filter_result = completion.model_extra["prompt_filter_results"][0]["content_filter_results"]
print("\nPrompt content filter results:")
for category, details in prompt_filter_result.items():
    print(f"{category}:\n filtered={details['filtered']}\n severity={details['severity']}")

# completion content filter result
print("\nCompletion content filter results:")
completion_filter_result = completion.choices[0].model_extra["content_filter_results"]
for category, details in completion_filter_result.items():
    print(f"{category}:\n filtered={details['filtered']}\n severity={details['severity']}")
