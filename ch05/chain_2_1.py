from llm_models import get_llm
from utilities import to_obj
from prompts import (
    WEB_SEARCH_PROMPT_TEMPLATE
)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

NUM_SEARCH_QUERIES = 2

web_searches_chain = (
    RunnableLambda(lambda x:
        {
            'assistant_instructions': x['assistant_instructions'],
            'num_search_queries': NUM_SEARCH_QUERIES,
            'user_question': x['user_question']
        }
    )
    | WEB_SEARCH_PROMPT_TEMPLATE | get_llm() | StrOutputParser() | to_obj 
)
