from llm_models import get_llm
from utilities import to_obj
from prompts import (
    WEB_SEARCH_PROMPT_TEMPLATE
)
from langchain.schema.output_parser import StrOutputParser

NUM_SEARCH_QUERIES = 2

web_searches_chain = (
    {
        'assistant_instructions': lambda x: x['assistant_instructions'],
        'num_search_queries': lambda x: NUM_SEARCH_QUERIES,
        'user_question': lambda x: x['user_question']
    }
    | WEB_SEARCH_PROMPT_TEMPLATE | get_llm() | StrOutputParser() | to_obj
)
