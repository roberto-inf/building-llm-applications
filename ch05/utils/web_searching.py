from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import List, Dict, Any
import time
import random
import requests
from duckduckgo_search.exceptions import DuckDuckGoSearchException

# Create a singleton instance of the DuckDuckGoSearchAPIWrapper to reuse
_ddg_instance = None

# Track the last request time to implement rate limiting
_last_request_time = 0
_min_request_interval = 2.0  # Minimum seconds between requests

def get_ddg_instance():
    """Get a singleton instance of DuckDuckGoSearchAPIWrapper."""
    global _ddg_instance
    if _ddg_instance is None:
        _ddg_instance = DuckDuckGoSearchAPIWrapper()
    return _ddg_instance

def web_search(web_query: str, num_results: int) -> List[str]:
    """
    Perform a web search with rate limiting, retry logic, and fallback mechanisms.
    
    Args:
        web_query: The search query
        num_results: Number of results to return
    
    Returns:
        List of URLs from search results
    """
    global _last_request_time
    
    # Implement rate limiting
    current_time = time.time()
    time_since_last_request = current_time - _last_request_time
    
    if time_since_last_request < _min_request_interval:
        # Wait to avoid hitting rate limits
        sleep_time = _min_request_interval - time_since_last_request + random.uniform(0.1, 0.5)
        print(f"Rate limiting: Waiting {sleep_time:.2f} seconds before next search request...")
        time.sleep(sleep_time)
    
    # Try DuckDuckGo with retry logic
    max_retries = 3
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Update the last request time
            _last_request_time = time.time()
            
            # Get the search results
            ddg = get_ddg_instance()
            results = ddg.results(web_query, num_results)
            
            # Extract the URLs
            urls = [r["link"] for r in results]
            
            # If we got results, return them
            if urls:
                return urls
            else:
                print(f"No results found for query: {web_query}")
                # If no results, try a fallback on the last attempt
                if attempt == max_retries - 1:
                    break
                time.sleep(1)  # Brief pause before retrying
                
        except DuckDuckGoSearchException as e:
            if "Ratelimit" in str(e) and attempt < max_retries - 1:
                # Exponential backoff with jitter for rate limit errors
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"DuckDuckGo rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"DuckDuckGo search failed: {str(e)}")
                break
        except Exception as e:
            print(f"Error during web search: {str(e)}")
            if attempt < max_retries - 1:
                # Simple retry for other errors
                delay = base_delay + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                break
    
    # If we get here, all attempts failed or returned no results
    # Use a fallback search mechanism
    return fallback_search(web_query, num_results)

def fallback_search(query: str, num_results: int) -> List[str]:
    """
    Fallback search mechanism when DuckDuckGo is rate-limited or fails.
    Returns a list of relevant Wikipedia and general knowledge URLs.
    
    Args:
        query: The search query
        num_results: Number of results to return
    
    Returns:
        List of URLs that might be relevant to the query
    """
    print(f"Using fallback search for query: {query}")
    
    # Clean and prepare the query
    query_clean = query.lower().strip()
    
    # List of general knowledge sources that cover a wide range of topics
    general_sources = [
        "https://en.wikipedia.org/wiki/Main_Page",
        "https://www.britannica.com/",
        "https://www.worldcat.org/",
        "https://www.jstor.org/",
        "https://www.sciencedirect.com/",
        "https://www.researchgate.net/",
        "https://scholar.google.com/",
        "https://www.academia.edu/"
    ]
    
    # Try to generate Wikipedia URLs based on key terms in the query
    wikipedia_urls = []
    
    # Extract potential Wikipedia topics from the query
    # Remove common question words and stop words
    stop_words = ["what", "where", "when", "why", "how", "is", "are", "was", "were", 
                 "do", "does", "did", "can", "could", "would", "should", "might",
                 "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
                 "to", "of", "from", "as", "tell", "me", "about", "you", "i", "we"]
    
    # Split the query into words and filter out stop words
    query_words = [word for word in query_clean.split() if word not in stop_words]
    
    # Generate potential Wikipedia URLs
    if len(query_words) >= 2:
        # Try pairs of words
        for i in range(len(query_words) - 1):
            topic = "_".join([query_words[i].capitalize(), query_words[i+1].capitalize()])
            wikipedia_urls.append(f"https://en.wikipedia.org/wiki/{topic}")
    
    # Add single word topics
    for word in query_words:
        if len(word) > 3:  # Only use meaningful words
            wikipedia_urls.append(f"https://en.wikipedia.org/wiki/{word.capitalize()}")
    
    # Try to create a direct Wikipedia search URL
    search_query = query_clean.replace(" ", "+")
    wikipedia_search_url = f"https://en.wikipedia.org/w/index.php?search={search_query}"
    wikipedia_urls.insert(0, wikipedia_search_url)
    
    # Combine Wikipedia URLs with general sources
    all_urls = wikipedia_urls + general_sources
    
    # Remove duplicates while preserving order
    unique_urls = []
    for url in all_urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    # Return the top N results
    return unique_urls[:num_results]