from models import get_llm, SearchQuery, SearchResult, SearchSummary
from prompts import WEB_SEARCH_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE
from utils.web_searching import web_search
from utils.web_scraping import web_scrape
import json
from typing import Dict, Any, List

NUM_SEARCH_QUERIES = 3
NUM_SEARCH_RESULTS_PER_QUERY = 3
RESULT_TEXT_MAX_CHARACTERS = 10000

def generate_search_queries(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate search queries based on the assistant instructions and user question.
    """
    assistant_info = state["assistant_info"]
    user_question = state["user_question"]
    assistant_instructions = assistant_info["assistant_instructions"]
    
    # Format the prompt
    prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
        assistant_instructions=assistant_instructions,
        user_question=user_question,
        num_search_queries=NUM_SEARCH_QUERIES
    )
    
    # Get the LLM response
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content
    
    # Parse the response to get the search queries
    try:
        # Extract the JSON array from the response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        json_str = response_text[json_start:json_end]
        
        # Parse the JSON
        search_queries = json.loads(json_str)
        
        # Return the updated state
        return {"search_queries": search_queries}
    except Exception as e:
        # Fallback to a default search query if parsing fails
        default_queries = [
            {"search_query": user_question, "user_question": user_question}
        ]
        return {"search_queries": default_queries}

def perform_web_searches(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform web searches based on the generated search queries.
    """
    search_queries = state["search_queries"]
    search_results = []
    
    # For each search query, get the search results
    for query_obj in search_queries:
        search_query = query_obj["search_query"]
        user_question = query_obj["user_question"]
        
        # Get the search results
        urls = web_search(web_query=search_query, num_results=NUM_SEARCH_RESULTS_PER_QUERY)
        
        # Add the results to the list
        for url in urls:
            search_results.append({
                "result_url": url,
                "search_query": search_query,
                "user_question": user_question
            })
    
    # Return the updated state
    return {"search_results": search_results}

def summarize_search_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize the search results.
    """
    search_results = state["search_results"]
    llm = get_llm()
    summaries = []
    
    # For each search result, get the text and summarize it
    for result in search_results:
        result_url = result["result_url"]
        search_query = result["search_query"]
        user_question = result["user_question"]
        
        try:
            # Get the webpage content
            search_result_text = web_scrape(url=result_url)[:RESULT_TEXT_MAX_CHARACTERS]
            
            # Format the prompt
            prompt = SUMMARY_PROMPT_TEMPLATE.format(
                search_result_text=search_result_text,
                search_query=search_query
            )
            
            # Get the summary
            summary_response = llm.invoke(prompt)
            text_summary = summary_response.content
            
            # Create the summary object
            summary = {
                "summary": f"Source Url: {result_url}\nSummary: {text_summary}",
                "result_url": result_url,
                "user_question": user_question
            }
            
            summaries.append(summary)
        except Exception as e:
            # Skip this result if there's an error
            continue
    
    # Create the research summary
    if summaries:
        research_summary = "\n\n".join([s["summary"] for s in summaries])
    else:
        research_summary = "No relevant information found."
    
    # Return the updated state
    return {
        "search_summaries": summaries,
        "research_summary": research_summary
    }
