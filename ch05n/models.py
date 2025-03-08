from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, TypedDict, Optional

openai_api_key = 'sk-8OycLFnI7Q49GzU8T5zchhxAA1vUUCk7VCDylQdWS1T3BlbkFJV8d16LKtYeTnOHY-BC2V2FFJHMb-YvbIDlI8FrIaYA'  # replace with your key

def get_llm():
    return ChatOpenAI(openai_api_key=openai_api_key,
                 model_name="gpt-4o-mini")

# Define typed dictionaries for state handling
class AssistantInfo(TypedDict):
    assistant_type: str
    assistant_instructions: str
    user_question: str

class SearchQuery(TypedDict):
    search_query: str
    user_question: str

class SearchResult(TypedDict):
    result_url: str
    search_query: str
    user_question: str

class SearchSummary(TypedDict):
    summary: str
    result_url: str
    user_question: str

class ResearchReport(TypedDict):
    report: str

# Graph state
class ResearchState(TypedDict):
    user_question: str
    assistant_info: Optional[AssistantInfo]
    search_queries: Optional[List[SearchQuery]]
    search_results: Optional[List[SearchResult]]
    search_summaries: Optional[List[SearchSummary]]
    research_summary: Optional[str]
    final_report: Optional[str]
