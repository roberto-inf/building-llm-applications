# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import asyncio
import operator
import os
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
from dotenv import load_dotenv


from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph_supervisor.supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage

from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import create_react_agent


# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------

load_dotenv() #A
#A load the environment variables from the .env 

# -----------------------------------------------------------------------------
# 1. Prepare knowledge base at startup
# -----------------------------------------------------------------------------

UK_DESTINATIONS = [ #A
    "Cornwall",
    "North_Cornwall",
    "South_Cornwall",
    "West_Cornwall",
]

async def build_vectorstore(destinations: Sequence[str]) -> Chroma: #B
    """Download WikiVoyage pages and create a Chroma vector store."""
    urls = [f"https://en.wikivoyage.org/wiki/{slug}" for slug in destinations] #C
    loader = AsyncHtmlLoader(urls) #C
    print("Downloading destination pages ...") #C
    docs = await loader.aload() #C

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128) #D
    chunks = sum([splitter.split_documents([d]) for d in docs], []) #D

    print(f"Embedding {len(chunks)} chunks ...") #E
    vectordb_client = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings()) #E
    print("Vector store ready.\n")
    return vectordb_client #F


# Singleton pattern (build once)
_ti_vectorstore_client: Chroma | None = None #G

def get_travel_info_vectorstore() -> Chroma: #H
    global _ti_vectorstore_client
    if _ti_vectorstore_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set the OPENAI_API_KEY env variable and re-run.")
        _ti_vectorstore_client = asyncio.run(build_vectorstore(UK_DESTINATIONS))
    return _ti_vectorstore_client #I

ti_vectorstore_client = get_travel_info_vectorstore() #J
ti_retriever = ti_vectorstore_client.as_retriever() #K

# A Destination list; you can add more destinations here
# B Function to build the vectorstore and return a reference to the vectorstore client
# C Load the destination pages asynchronously from the web into a list of documents
# D Split the documents into chunks of 1024 characters with 128 characters of overlap    
# E Embed the chunks and store them in the vectorstore
# F Return the vectorstore client
# G Initialize a cache for the vectorstore client instance as None
# H Function to trigger the creation of the vectorstore and return a reference to the cache of its client instance
# I Return the a reference to the cache of the vectorstore client instance
# J Instantiate the vectorstore client
# K Instantiate the vectorstore retriever


# ----------------------------------------------------------------------------
# 2. Define the only tool
# ----------------------------------------------------------------------------

@tool(description="Search travel information about destinations in England.") #A
def search_travel_info(query: str) -> str: #B
    """Search embedded WikiVoyage content for information about destinations in England."""
    docs = ti_retriever.invoke(query) #C
    top = docs[:4] if isinstance(docs, list) else docs #C
    return "\n---\n".join(d.page_content for d in top) #D

#A Define the tool using the @tool decorator
#B Define the tool function, which takes a query, performs a semantic search and returns a string response from the vectorstore
#C Perform a semantic search on the vectorstore and return the top 4 results
#D Joins the top 4 results into a single string

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------

async def get_accuweather_tools(): #A
    mcp_client = MultiServerMCPClient({ #B
        "accuweather": { #C
            "url": "http://127.0.0.1:8020/accu-mcp-server",
            "transport": "streamable_http"
        }
    })
    return await mcp_client.get_tools() #D

#A Define the function to get the AccuWeather tools as an async function
#B Instantiate the MultiServerMCPClient
#C Register the AccuWeather MCP server
#D Return the AccuWeather tools exposed by the MCP server


async def chat_loop(agent): #A
    print("UK Travel Assistant (type 'exit' to quit)")
    while True: #B
        user_input = input("You: ").strip() #C
        if user_input.lower() in {"exit", "quit"}: #D
            break
        state = {"messages": [HumanMessage(
            content=user_input)]} #E
        result = await agent.ainvoke(state) #F
        response_msg = result["messages"][-1] #G
        print(
           f"Assistant: {response_msg.content}\n") #H

#A Define the chat loop as an async function
#B Start the chat loop
#C Get the user input
#D Check if the user input is "exit" or "quit" to exit the loop
#E Create the initial state with a HumanMessage containing the user input
#F Invoke the agent with the initial state, asyncronously
#G Get the last message from the result, which contains the final answer
#H Print the assistant's final answer, from the content of the last message

class AgentState(TypedDict): #A
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps

async def main():
    accuweather_tools = \
        await get_accuweather_tools() #B
    tools = [search_travel_info, 
        *accuweather_tools] #C
    llm_model = ChatOpenAI( 
        model="gpt-5-mini",
        use_responses_api=True) #D

    travel_info_agent = create_react_agent( #E
        model=llm_model,
        tools=tools,
        state_schema=AgentState,
        name="travel_info_agent",
        prompt="""You are a helpful assistant that can 
        search travel information and get the weather forecast. 
        Only use the tools to find the information you need 
        (including town names).""",
    )
    await chat_loop(travel_info_agent) #F

if __name__ == "__main__":
    asyncio.run(main()) #G

#A - Define the AgentState class
#B - Get the AccuWeather MCP server tools
#C - Combine the local search_travel_info tool with the AccuWeather MCP server tools
#D - Instantiate the LLM model
#E - Create the travel_info_agent
#F - Start the chat loop
#G - Run the main function, asyncronously      