
from utilities import to_obj
from web_searching import web_search
from langchain.schema.runnable import RunnableLambda

NUM_SEARCH_RESULTS_PER_QUERY = 3

search_result_urls_chain = (
    RunnableLambda(lambda x: 
        [
            {
                'result_url': url, 
                'search_query': x['search_query'],
                'user_question': x['user_question']
            }
            for url in web_search(web_query=x['search_query'], 
                                  num_results=NUM_SEARCH_RESULTS_PER_QUERY)
        ]
    )
)