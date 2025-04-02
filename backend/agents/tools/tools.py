import math
import re

import numexpr
from langchain_core.tools import BaseTool, tool
import os 
# import openai
# import pinecone
import time


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # openai embed model
# embed_model_id = "text-embedding-ada-002"


# # create openai embed model
# res = openai.Embedding.create(
#     input=[
#         "We would have some text to embed here",
#         "And maybe another chunk here too"
#     ], engine=embed_model_id
# )

# # PINECONE 
# api_key = os.getenv("PINECONE_API_KEY")
# pinecone.init(api_key=api_key)

# index_name = "cfa-research" # double check the index name in the Pinecone dashboard

# # check if index already exists (it shouldn't if this is first time)
# if index_name not in pinecone.list_indexes():
#     # if does not exist, create index
#     pinecone.create_index(
#         index_name,
#         dimension=len(res['data'][0]['embedding']),
#         metric='cosine'
#     )
#     # wait for index to be initialized
#     while not pinecone.describe_index(index_name).status['ready']:
#         time.sleep(1)

# # connect to index
# index = pinecone.Index(index_name)


# # pinecone retrieve tool
# @tool
# async def retrieve(query: str) -> list:
#     # create query embedding
#     res = openai.Embedding.create(input=[query], engine=embed_model_id)
#     xq = res['data'][0]['embedding']
#     # get relevant contexts from pinecone
#     res = index.query(xq, top_k=5, include_metadata=True)
#     # get list of retrieved texts
#     contexts = [x['metadata']['chunk'] for x in res['matches']]
#     return contexts

def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"

