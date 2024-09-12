
from llm_models import get_llm
from web_scraping import web_scrape
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from prompts import (
    SUMMARY_PROMPT_TEMPLATE
)

RESULT_TEXT_MAX_CHARACTERS = 10000

search_result_text_and_summary_chain = (
    RunnableLambda(lambda x:
        {
            'search_result_text': web_scrape(url=x['result_url'])[:RESULT_TEXT_MAX_CHARACTERS],
            'result_url': x['result_url'], 
            'search_query': x['search_query'],
            'user_question': x['user_question']
        }
    )
    | RunnableParallel (
        {
            'text_summary': SUMMARY_PROMPT_TEMPLATE | get_llm() | StrOutputParser(),
            'result_url': lambda x: x['result_url'],
            'user_question': lambda x: x['user_question']            
        }
    )
    | RunnableLambda(lambda x: 
        {
            'summary': f"Source Url: {x['result_url']}\nSummary: {x['text_summary']}",
            'user_question': x['user_question']
        }
    ) 
)