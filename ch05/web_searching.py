from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import List

def web_search(web_query: str, num_results: int) -> List[str]:
    return [r["link"] for r in DuckDuckGoSearchAPIWrapper().results(web_query, num_results)]