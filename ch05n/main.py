from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List, Annotated, Tuple
import os

from models import ResearchState
from agents.assistant_selector import select_assistant
from agents.web_researcher import generate_search_queries, perform_web_searches, summarize_search_results
from agents.report_writer import write_research_report

def create_research_graph() -> StateGraph:
    """
    Create the LangGraph research graph that coordinates the agents.
    """
    # Define the graph
    graph = StateGraph(ResearchState)
    
    # Add nodes to the graph
    graph.add_node("select_assistant", select_assistant)
    graph.add_node("generate_search_queries", generate_search_queries)
    graph.add_node("perform_web_searches", perform_web_searches)
    graph.add_node("summarize_search_results", summarize_search_results)
    graph.add_node("write_research_report", write_research_report)
    
    # Define the flow of the graph
    graph.add_edge("select_assistant", "generate_search_queries")
    graph.add_edge("generate_search_queries", "perform_web_searches")
    graph.add_edge("perform_web_searches", "summarize_search_results")
    graph.add_edge("summarize_search_results", "write_research_report")
    graph.add_edge("write_research_report", END)
    
    # Set the entry point
    graph.set_entry_point("select_assistant")
    
    return graph

def run_research(question: str) -> str:
    """
    Run the research graph with a user question.
    
    Args:
        question: The user's research question
        
    Returns:
        The final research report
    """
    # Create the graph
    research_graph = create_research_graph()
    
    # Compile the graph
    app = research_graph.compile()
    
    # Initialize the state
    initial_state = {
        "user_question": question,
        "assistant_info": None,
        "search_queries": None,
        "search_results": None,
        "search_summaries": None,
        "research_summary": None,
        "final_report": None
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    # Extract and return the final report
    return result["final_report"]

# For testing purposes
if __name__ == "__main__":
    # Example usage
    question = "What are the latest advancements in renewable energy?"
    report = run_research(question)
    print(report)
