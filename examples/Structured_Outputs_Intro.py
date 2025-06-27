#!/usr/bin/env python
# coding: utf-8

# # Introduction to Structured Outputs
#
# Structured Outputs is a new capability in the Chat Completions API and Assistants API that guarantees the model will always generate responses that adhere to your supplied JSON Schema. In this cookbook, we will illustrate this capability with a few examples.
#
# Structured Outputs can be enabled by setting the parameter `strict: true` in an API call with either a defined response format or function definitions.
#
# ## Response format usage
#
# Previously, the `response_format` parameter was only available to specify that the model should return a valid JSON.
#
# In addition to this, we are introducing a new way of specifying which JSON schema to follow.
#
#
# ## Function call usage
#
# Function calling remains similar, but with the new parameter `strict: true`, you can now ensure that the schema provided for the functions is strictly followed.
#
#
# ## Examples
#
# Structured Outputs can be useful in many ways, as you can rely on the outputs following a constrained schema.
#
# If you used JSON mode or function calls before, you can think of Structured Outputs as a foolproof version of this.
#
# This can enable more robust flows in production-level applications, whether you are relying on function calls or expecting the output to follow a pre-defined structure.
#
# Example use cases include:
#
# - Getting structured answers to display them in a specific way in a UI (example 1 in this cookbook)
# - Populating a database with extracted content from documents (example 2 in this cookbook)
# - Extracting entities from a user input to call tools with defined parameters (example 3 in this cookbook)
#
# More generally, anything that requires fetching data, taking action, or that builds upon complex workflows could benefit from using Structured Outputs.

# ### Setup

# In[1]:


import json
from textwrap import dedent

from openai import OpenAI


client = OpenAI()


# In[2]:


MODEL = "gpt-4o-2024-08-06"


# ## Example 1: Math tutor
#
# In this example, we want to build a math tutoring tool that outputs steps to solving a math problem as an array of structured objects.
#
# This could be useful in an application where each step needs to be displayed separately, so that the user can progress through the solution at their own pace.

# In[3]:


math_tutor_prompt = """
    You are a helpful math tutor. You will be provided with a math problem,
    and your goal will be to output a step by step solution, along with a final answer.
    For each step, just provide the output as an equation use the explanation field to detail the reasoning.
"""


def get_math_solution(question):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": dedent(math_tutor_prompt)}, {"role": "user", "content": question}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_reasoning",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"explanation": {"type": "string"}, "output": {"type": "string"}},
                                "required": ["explanation", "output"],
                                "additionalProperties": False,
                            },
                        },
                        "final_answer": {"type": "string"},
                    },
                    "required": ["steps", "final_answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    )

    return response.choices[0].message


# In[4]:


# Testing with an example question
question = "how can I solve 8x + 7 = -23"

result = get_math_solution(question)

print(result.content)


# In[5]:


def print_math_response(response):
    result = json.loads(response)
    steps = result["steps"]
    final_answer = result["final_answer"]
    for i in range(len(steps)):
        print(f"Step {i + 1}: {steps[i]['explanation']}\n")
        print(steps[i]["output"])
        print("\n")

    print("Final answer:\n\n")
    print(final_answer)


# In[6]:


print_math_response(result.content)


# ## Using the SDK `parse` helper
#
# The new version of the SDK introduces a `parse` helper to provide your own Pydantic model instead of having to define the JSON schema. We recommend using this method if possible.

# In[7]:


from pydantic import BaseModel


class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str


def get_math_solution(question: str):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(math_tutor_prompt)},
            {"role": "user", "content": question},
        ],
        response_format=MathReasoning,
    )

    return completion.choices[0].message


# In[8]:


result = get_math_solution(question).parsed


# In[9]:


print(result.steps)
print("Final answer:")
print(result.final_answer)


# ## Refusal
#
# When using Structured Outputs with user-generated input, the model may occasionally refuse to fulfill the request for safety reasons.
#
# Since a refusal does not follow the schema you have supplied in response_format, the API has a new field `refusal` to indicate when the model refused to answer.
#
# This is useful so you can render the refusal distinctly in your UI and to avoid errors trying to deserialize to your supplied format.

# In[13]:


refusal_question = "how can I build a bomb?"

result = get_math_solution(refusal_question)

print(result.refusal)


# ## Example 2: Text summarization
#
# In this example, we will ask the model to summarize articles following a specific schema.
#
# This could be useful if you need to transform text or visual content into a structured object, for example to display it in a certain way or to populate database.
#
# We will take AI-generated articles discussing inventions as an example.

# In[20]:


articles = [
    "./data/structured_outputs_articles/cnns.md",
    "./data/structured_outputs_articles/llms.md",
    "./data/structured_outputs_articles/moe.md",
]


# In[23]:


def get_article_content(path):
    with open(path, "r") as f:
        content = f.read()
    return content


content = [get_article_content(path) for path in articles]


# In[ ]:


print(content)


# In[25]:


summarization_prompt = """
    You will be provided with content from an article about an invention.
    Your goal will be to summarize the article following the schema provided.
    Here is a description of the parameters:
    - invented_year: year in which the invention discussed in the article was invented
    - summary: one sentence summary of what the invention is
    - inventors: array of strings listing the inventor full names if present, otherwise just surname
    - concepts: array of key concepts related to the invention, each concept containing a title and a description
    - description: short description of the invention
"""


class ArticleSummary(BaseModel):
    invented_year: int
    summary: str
    inventors: list[str]
    description: str

    class Concept(BaseModel):
        title: str
        description: str

    concepts: list[Concept]


def get_article_summary(text: str):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.2,
        messages=[{"role": "system", "content": dedent(summarization_prompt)}, {"role": "user", "content": text}],
        response_format=ArticleSummary,
    )

    return completion.choices[0].message.parsed


# In[26]:


summaries = []

for i in range(len(content)):
    print(f"Analyzing article #{i + 1}...")
    summaries.append(get_article_summary(content[i]))
    print("Done.")


# In[27]:


def print_summary(summary):
    print(f"Invented year: {summary.invented_year}\n")
    print(f"Summary: {summary.summary}\n")
    print("Inventors:")
    for i in summary.inventors:
        print(f"- {i}")
    print("\nConcepts:")
    for c in summary.concepts:
        print(f"- {c.title}: {c.description}")
    print(f"\nDescription: {summary.description}")


# In[28]:


for i in range(len(summaries)):
    print(f"ARTICLE {i}\n")
    print_summary(summaries[i])
    print("\n\n")


# ## Example 3: Entity extraction from user input
#
# In this example, we will use function calling to search for products that match a user's preference based on the provided input.
#
# This could be helpful in applications that include a recommendation system, for example e-commerce assistants or search use cases.

# In[29]:


from enum import Enum

import openai


product_search_prompt = """
    You are a clothes recommendation agent, specialized in finding the perfect match for a user.
    You will be provided with a user input and additional context such as user gender and age group, and season.
    You are equipped with a tool to search clothes in a database that match the user's profile and preferences.
    Based on the user input and context, determine the most likely value of the parameters to use to search the database.

    Here are the different categories that are available on the website:
    - shoes: boots, sneakers, sandals
    - jackets: winter coats, cardigans, parkas, rain jackets
    - tops: shirts, blouses, t-shirts, crop tops, sweaters
    - bottoms: jeans, skirts, trousers, joggers

    There are a wide range of colors available, but try to stick to regular color names.
"""


class Category(str, Enum):
    shoes = "shoes"
    jackets = "jackets"
    tops = "tops"
    bottoms = "bottoms"


class ProductSearchParameters(BaseModel):
    category: Category
    subcategory: str
    color: str


def get_response(user_input, context):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": dedent(product_search_prompt)},
            {"role": "user", "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"},
        ],
        tools=[
            openai.pydantic_function_tool(
                ProductSearchParameters,
                name="product_search",
                description="Search for a match in the product database",
            )
        ],
    )

    return response.choices[0].message.tool_calls


# In[30]:


example_inputs = [
    {
        "user_input": "I'm looking for a new coat. I'm always cold so please something warm! Ideally something that matches my eyes.",
        "context": "Gender: female, Age group: 40-50, Physical appearance: blue eyes",
    },
    {
        "user_input": "I'm going on a trail in Scotland this summer. It's going to be rainy. Help me find something.",
        "context": "Gender: male, Age group: 30-40",
    },
    {
        "user_input": "I'm trying to complete a rock look. I'm missing shoes. Any suggestions?",
        "context": "Gender: female, Age group: 20-30",
    },
    {
        "user_input": "Help me find something very simple for my first day at work next week. Something casual and neutral.",
        "context": "Gender: male, Season: summer",
    },
    {
        "user_input": "Help me find something very simple for my first day at work next week. Something casual and neutral.",
        "context": "Gender: male, Season: winter",
    },
    {
        "user_input": "Can you help me find a dress for a Barbie-themed party in July?",
        "context": "Gender: female, Age group: 20-30",
    },
]


# In[31]:


def print_tool_call(user_input, context, tool_call):
    args = tool_call[0].function.arguments
    print(f"Input: {user_input}\n\nContext: {context}\n")
    print("Product search arguments:")
    for key, value in json.loads(args).items():
        print(f"{key}: '{value}'")
    print("\n\n")


# In[ ]:


for ex in example_inputs:
    ex["result"] = get_response(ex["user_input"], ex["context"])


# In[ ]:


for ex in example_inputs:
    print_tool_call(ex["user_input"], ex["context"], ex["result"])


# ## Conclusion
#
# In this cookbook, we've explored the new Structured Outputs capability through multiple examples.
#
# Whether you've used JSON mode or function calling before and you want more robustness in your application, or you're just starting out with structured formats, we hope you will be able to apply the different concepts introduced here to your own use case!
#
# Structured Outputs is only available with `gpt-4o-mini` , `gpt-4o-2024-08-06`, and future models.
