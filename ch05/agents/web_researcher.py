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
    Uses different strategies based on iteration count to ensure variety.
    """
    assistant_info = state["assistant_info"]
    user_question = state["user_question"]
    assistant_instructions = assistant_info["assistant_instructions"]
    
    # Get the current iteration count
    iteration_count = state.get("iteration_count", 0)
    
    # Check if this is a regeneration (we already have search queries and relevance evaluation)
    previous_queries = state.get("search_queries", [])
    relevance_evaluation = state.get("relevance_evaluation", None)
    
    # Format the prompt based on iteration count
    if iteration_count == 0:
        # First-time query generation
        print("Generating initial search queries...")
        prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
            assistant_instructions=assistant_instructions,
            user_question=user_question,
            num_search_queries=NUM_SEARCH_QUERIES
        )
    elif iteration_count == 1:
        # Second iteration - more specific queries
        print("First regeneration: Creating more specific queries...")
        previous_query_list = ", ".join([q["search_query"] for q in previous_queries])
        relevance_percentage = relevance_evaluation.get("relevance_percentage", 0) if relevance_evaluation else 0
        relevance_explanation = relevance_evaluation.get("explanation", "No explanation provided") if relevance_evaluation else ""
        
        prompt = f"""
        {assistant_instructions}

        You are generating new search queries because the previous queries did not yield sufficiently relevant results.
        
        Original question: {user_question}
        
        Previous search queries: {previous_query_list}
        
        Relevance evaluation: {relevance_percentage}% relevant
        Explanation: {relevance_explanation}
        
        Please generate {NUM_SEARCH_QUERIES} NEW and DIFFERENT web search queries that are MORE SPECIFIC and TARGETED 
        to gather relevant information on the original question. 
        
        IMPORTANT: DO NOT repeat or rephrase the previous queries. Create completely different approaches to finding information.
        
        You must respond with a list of queries in the following format:
        [
            {{"search_query": "query1", "user_question": "{user_question}" }},
            {{"search_query": "query2", "user_question": "{user_question}" }},
            {{"search_query": "query3", "user_question": "{user_question}" }}
        ]
        """
    else:
        # Third or later iteration - completely different approach
        print(f"Iteration {iteration_count}: Using alternative search strategies...")
        all_previous_queries = ", ".join([q["search_query"] for q in previous_queries])
        
        prompt = f"""
        {assistant_instructions}

        You are generating search queries for the FINAL attempt to find relevant information.
        
        Original question: {user_question}
        
        All previous search queries that DID NOT yield relevant results: {all_previous_queries}
        
        For this final attempt, take a completely different angle. Consider:
        1. Breaking down the question into smaller, more focused sub-questions
        2. Using technical or specialized terms related to the topic
        3. Searching for expert opinions or academic perspectives
        4. Looking for case studies or specific examples
        5. Exploring historical context or background information
        
        CRITICAL INSTRUCTIONS:
        1. DO NOT repeat or rephrase ANY previous queries listed above
        2. Generate queries that are COMPLETELY DIFFERENT from all previous attempts
        
        Please generate {NUM_SEARCH_QUERIES} COMPLETELY NEW search queries following the strategy above.
        
        You must respond with a list of queries in the following format:
        [
            {{"search_query": "query1", "user_question": "{user_question}" }},
            {{"search_query": "query2", "user_question": "{user_question}" }},
            {{"search_query": "query3", "user_question": "{user_question}" }}
        ]
        """
    
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
        
        print(f"Generated {len(search_queries)} search queries")
        for i, query in enumerate(search_queries):
            print(f"  Query {i+1}: {query['search_query']}")
        
        # Return the updated state
        return {
            "search_queries": search_queries,
            # Reset the relevance evaluation and regeneration flag when generating new queries
            "relevance_evaluation": None,
            "should_regenerate_queries": None
        }
    except Exception as e:
        print(f"Error parsing search queries: {str(e)}")
        # Fallback to a default search query if parsing fails
        default_queries = [
            {"search_query": f"{user_question} iteration {iteration_count + 1}", "user_question": user_question}
        ]
        print(f"Using default query: {default_queries[0]['search_query']}")
        return {
            "search_queries": default_queries,
            "relevance_evaluation": None,
            "should_regenerate_queries": None
        }

def perform_web_searches(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform web searches based on the generated search queries.
    """
    search_queries = state["search_queries"]
    search_results = []
    fallback_used = False
    
    print(f"Performing web searches for {len(search_queries)} queries...")
    
    # For each search query, get the search results
    for query_obj in search_queries:
        search_query = query_obj["search_query"]
        user_question = query_obj["user_question"]
        
        try:
            # Get the search results
            print(f"Searching for: {search_query}")
            urls = web_search(web_query=search_query, num_results=NUM_SEARCH_RESULTS_PER_QUERY)
            
            # Check if these are likely fallback results (Wikipedia URLs)
            if any("wikipedia.org" in url for url in urls[:2]):
                print(f"Fallback search was used for query: {search_query}")
                fallback_used = True
                is_fallback = True
            else:
                is_fallback = False
            
            # Add the results to the list
            for url in urls:
                search_results.append({
                    "result_url": url,
                    "search_query": search_query,
                    "user_question": user_question,
                    "is_fallback": is_fallback
                })
                
            print(f"Found {len(urls)} results for query: {search_query}")
        except Exception as e:
            print(f"Error searching for '{search_query}': {str(e)}")
            # Continue with other queries even if one fails
            continue
    
    # If we have no search results at all, add a fallback result
    if not search_results:
        print("No search results found. Using general fallback information.")
        fallback_url = "https://en.wikipedia.org/wiki/Main_Page"
        search_results.append({
            "result_url": fallback_url,
            "search_query": "general information",
            "user_question": state["user_question"],
            "is_fallback": True
        })
        fallback_used = True
    
    # Return the updated state with information about fallback usage
    return {
        "search_results": search_results,
        "used_fallback_search": fallback_used
    }

def summarize_search_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize the search results.
    """
    search_results = state["search_results"]
    used_fallback_search = state.get("used_fallback_search", False)
    llm = get_llm()
    summaries = []
    
    print(f"Summarizing {len(search_results)} search results...")
    
    # For each search result, get the text and summarize it
    for result in search_results:
        result_url = result["result_url"]
        search_query = result["search_query"]
        user_question = result["user_question"]
        is_fallback = result.get("is_fallback", False)
        
        try:
            # Get the webpage content
            print(f"Scraping content from: {result_url}")
            search_result_text = web_scrape(url=result_url)[:RESULT_TEXT_MAX_CHARACTERS]
            
            # Skip if the content is an error message or too short
            if search_result_text.startswith("Failed to") or len(search_result_text) < 50:
                print(f"Skipping {result_url} due to scraping issues or insufficient content")
                continue
            
            # Format the prompt, with additional context for fallback results
            if is_fallback:
                prompt = f"""
                You are summarizing content from a fallback source that was used because the primary search engine was unavailable.
                
                Read the following text:
                Text: {search_result_text} 
                
                -----------
                
                Using the above text, answer in short the following question.
                Question: {search_query}
                
                -----------
                If you cannot answer the question above using the text provided above, then just summarize the text. 
                Include all factual information, numbers, stats etc if available.
                
                Note that this is a fallback source, so it might not directly address the question.
                """
            else:
                prompt = SUMMARY_PROMPT_TEMPLATE.format(
                    search_result_text=search_result_text,
                    search_query=search_query
                )
            
            # Get the summary
            summary_response = llm.invoke(prompt)
            text_summary = summary_response.content
            
            # Add a note about fallback sources
            if is_fallback:
                source_note = "[Note: This information comes from a fallback source and may not directly address the question.]"
                text_summary = f"{text_summary}\n{source_note}"
            
            # Create the summary object
            summary = {
                "summary": f"Source Url: {result_url}\nSummary: {text_summary}",
                "result_url": result_url,
                "user_question": user_question,
                "is_fallback": is_fallback
            }
            
            summaries.append(summary)
            print(f"Successfully summarized content from: {result_url}")
        except Exception as e:
            print(f"Error summarizing {result_url}: {str(e)}")
            # Skip this result if there's an error
            continue
    
    # Create the research summary
    if summaries:
        research_summary = "\n\n".join([s["summary"] for s in summaries])
        print(f"Created research summary with {len(summaries)} sources")
        
        # Add a note if fallback search was used
        if used_fallback_search:
            fallback_note = "\n\n[Note: Some or all of this information comes from fallback sources because the primary search engine was unavailable. The information may not be as directly relevant to your question as usual.]"
            research_summary += fallback_note
    else:
        research_summary = "No relevant information found. Please try different search queries."
        print("Warning: No summaries were generated from search results")
    
    # Return the updated state
    return {
        "search_summaries": summaries,
        "research_summary": research_summary,
        "used_fallback_search": used_fallback_search
    }

def evaluate_search_relevance(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the relevance of search summaries to the original question.
    If less than 50% of summaries are relevant, return to search query generation.
    """
    search_summaries = state.get("search_summaries", [])
    user_question = state["user_question"]
    research_summary = state.get("research_summary", "")
    used_fallback_search = state.get("used_fallback_search", False)
    
    print("Evaluating relevance of search summaries to the original question...")
    
    # If there are no summaries, we need to regenerate queries
    if not search_summaries or not research_summary:
        print("No search summaries found. Regenerating search queries...")
        return {"should_regenerate_queries": True}
    
    # Use the LLM to evaluate relevance
    llm = get_llm()
    
    # Create a prompt for the LLM to evaluate relevance
    evaluation_prompt = f"""
    You are an expert research evaluator. Your task is to evaluate the relevance of search results 
    to the original research question.
    
    Original research question: {user_question}
    
    Search result summaries:
    {research_summary}
    
    For each search result summary, determine if it is relevant to answering the original question.
    Then calculate what percentage of the search results are relevant.
    
    Return your evaluation as a JSON object with the following structure:
    {{
        "relevance_percentage": <percentage of relevant results as a number between 0 and 100>,
        "explanation": <brief explanation of your evaluation>,
        "relevant_count": <number of relevant summaries>,
        "total_count": <total number of summaries>
    }}
    """
    
    try:
        # Get the evaluation from the LLM
        evaluation_response = llm.invoke(evaluation_prompt)
        evaluation_text = evaluation_response.content
        
        # Extract the JSON from the response
        try:
            # Find JSON in the response
            json_start = evaluation_text.find('{')
            json_end = evaluation_text.rfind('}') + 1
            json_str = evaluation_text[json_start:json_end]
            
            # Parse the JSON
            evaluation = json.loads(json_str)
            relevance_percentage = evaluation.get("relevance_percentage", 0)
            
            # Determine if we should regenerate queries (less than 50% relevant)
            should_regenerate = relevance_percentage < 50
            
            if should_regenerate:
                print(f"Only {relevance_percentage}% of search results are relevant. Regenerating search queries...")
            else:
                print(f"{relevance_percentage}% of search results are relevant. Proceeding to write research report...")
            
            return {
                "relevance_evaluation": evaluation,
                "should_regenerate_queries": should_regenerate
            }
        except Exception as e:
            print(f"Error parsing relevance evaluation: {str(e)}")
            # If we can't parse the evaluation, assume we need to regenerate
            return {"should_regenerate_queries": True}
    except Exception as e:
        print(f"Error during relevance evaluation: {str(e)}")
        # If there's an error, assume we need to regenerate
        return {"should_regenerate_queries": True}
