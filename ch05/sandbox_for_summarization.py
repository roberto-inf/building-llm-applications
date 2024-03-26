
items = [
            {'summary':"blah blah blah1", "user_question": "what's up?"},
            {'summary':"blah blah blah2", "user_question": "what's up?"},
            {'summary':"blah blah blah3", "user_question": "what's up?"},
            {'summary':"blah blah blah4", "user_question": "what's up?"},
            {'summary':"blah blah blah5", "user_question": "what's up?"},
            {'summary':"blah blah blah7", "user_question": "what's up?"}
        ]

def merge(x):
    return {'summary': '\n'.join([i['summary'] for i in x]), 'user_question': x[0]['user_question']}

print(merge(items))

from langchain.prompts import PromptTemplate

##################################
prompt_template = PromptTemplate.from_template(
    template="""This is a template {user_question}""",
    #input_variables=['user_question']
)

prompt = prompt_template.format(user_question="what's up?")


#### Web search
from langchain.utilities import DuckDuckGoSearchAPIWrapper
NUM_SEARCH_RESULTS = 1
question = 'What can I see and do in the Spanish town Astorga'

###
search_engine = DuckDuckGoSearchAPIWrapper()

# try --- DO NOT INCLUDE
# results = search_engine.results(question, NUM_SEARCH_RESULTS)
# print(results)

###
def web_search(query: str, num_results: int):
    return [r["link"] for r in search_engine.results(query, num_results)]

#try --- DO NOT INCLUDE
# search_results = web_search(query=question, num_results=NUM_SEARCH_RESULTS)
# print(search_results)
####

#### Web scrape
import requests
from bs4 import BeautifulSoup

def web_scrape(url: str):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"
    
## try --- DO NOT INCLUDE
# url = 'https://budtravelagency.com/things-to-do-in-astorga-spain/'
# scraped_text = web_scrape(url=url)
# print(scraped_text)

####
ASSISTAINT_SELECTION_INSTRUCTIONS = """
This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific assistaint, defined by its type and role, with each assistaint requiring distinct instructions.
Assistaint
The assistaint is determined by the field of the topic and the specific name of the assistaint that could be utilized to research the topic provided. Assistaints are categorized by their area of expertise, and each assistaint type is associated with a corresponding emoji.

examples:
task: "should I invest in apple stocks?"
response: 

{
    "assistaint_type": "Finance Assistaint",
    "assistaint_prompt": "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
}
task: "what are the most interesting sites in Tel Aviv?"
response:
{
    "assistaint_type": "Travel Assistaint",
    "assistaint_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
}
task: "Is Messi a good soccer player?"
response:
{
    "assistaint_type": "Sport Assistaint",
    "assistaint_prompt": "You are an experienced AI sport assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured sport reports on given sport personalities, or sport events, including factual details, statistics and insights."
}

------
This is the research task you should undertake:

task: {task}
""" 

####
from langchain.prompts import PromptTemplate
ASSISTAINT_SELECTION_PROMPT_TEMPLATE = PromptTemplate.from_template(
    ASSISTAINT_SELECTION_INSTRUCTIONS
)

WEB_SEARCH_INSTRUCTIONS = """
{assistaint_prompt}

Write 3 web search queries to gather as much information as possible 
on the following question: {question}. Your objective is to write a report based on the information you find.
You must respond with a list of strings in the following format: 
["query 1", "query 2", "query 3"].
"""

WEB_SEARCH_PROMPT_TEMPLATE = PromptTemplate.from_template(
    WEB_SEARCH_INSTRUCTIONS
)

SUMMARY_INSTRUCTIONS = """
Read the following text:Text: {text} 

-----------

Using the above text, answer in short the following question.
Question: {question}
 
-----------
If you cannot answer the question above using the text provided above, then just summarize the text. 
Include all factual information, numbers, stats etc if available.""" 
SUMMARY_PROMPT_TEMPLATE = PromptTemplate.from_template(
    SUMMARY_INSTRUCTIONS
)

# try simple 
    
