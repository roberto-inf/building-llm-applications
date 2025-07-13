# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import asyncio
import operator
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

# UK_DESTINATIONS = [ #A
#     "Cornwall",
#     "North_Cornwall",
#     "South_Cornwall",
#     "West_Cornwall",
# ]

# async def build_vectorstore(destinations: Sequence[str]) -> Chroma: #B
#     """Download WikiVoyage pages and create a Chroma vector store."""
#     urls = [f"https://en.wikivoyage.org/wiki/{slug}" for slug in destinations] #C
#     loader = AsyncHtmlLoader(urls) #C
#     print("Downloading destination pages ...") #C
#     docs = await loader.aload() #C

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128) #D
#     chunks = sum([splitter.split_documents([d]) for d in docs], []) #D

#     print(f"Embedding {len(chunks)} chunks ...") #E
#     vectordb_client = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings()) #E
#     print("Vector store ready.\n")
#     return vectordb_client #F


# # Singleton pattern (build once)
# _ti_vectorstore_client: Chroma | None = None #G

# def get_travel_info_vectorstore() -> Chroma: #H
#     global _ti_vectorstore_client
#     if _ti_vectorstore_client is None:
#         if not os.environ.get("OPENAI_API_KEY"):
#             raise RuntimeError("Set the OPENAI_API_KEY env variable and re-run.")
#         _ti_vectorstore_client = asyncio.run(build_vectorstore(UK_DESTINATIONS))
#     return _ti_vectorstore_client #I

# ti_vectorstore_client = get_travel_info_vectorstore() #J
# ti_retriever = ti_vectorstore_client.as_retriever() #K

#A Destination list; you can add more destinations here
#B Function to build the vectorstore and return a reference to the vectorstore client
#C Load the destination pages asynchronously from the web into a list of documents
#D Split the documents into chunks of 1024 characters with 128 characters of overlap    
#E Embed the chunks and store them in the vectorstore
#F Return the vectorstore client
#G Initialize a cache for the vectorstore client instance as None
#H Function to trigger the creation of the vectorstore and return a reference to the cache of its client instance
#I Return the a reference to the cache of the vectorstore client instance
#J Instantiate the vectorstore client
#K Instantiate the vectorstore retriever


# ----------------------------------------------------------------------------
# 2. Define the only tool
# ----------------------------------------------------------------------------

# @tool(description="Search travel information about destinations in England.") #A
# def search_travel_info(query: str) -> str: #B
#     """Search embedded WikiVoyage content for information about destinations in England."""
#     docs = ti_retriever.invoke(query) #C
#     top = docs[:4] if isinstance(docs, list) else docs #C
#     return "\n---\n".join(d.page_content for d in top) #D

#A Define the tool using the @tool decorator
#B Define the tool function, which takes a query, performs a semantic search and returns a string response from the vectorstore
#C Perform a semantic search on the vectorstore and return the top 4 results
#D Joins the top 4 results into a single string

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------

async def get_accuweather_tools():
    mcp_client = MultiServerMCPClient({
        "accuweather": {
            "url": "http://127.0.0.1:8020/accu-mcp-server",
            "transport": "streamable_http"
        }
    })
    return await mcp_client.get_tools()

async def chat_loop(agent):
    from langchain_core.messages import HumanMessage
    print("UK Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        state = {"messages": [HumanMessage(content=user_input)]}
        result = await agent.ainvoke(state)
        response_msg = result["messages"][-1]
        print(f"Assistant: {response_msg.content}\n")

async def main():
    accuweather_tools = await get_accuweather_tools()
    tools = [*accuweather_tools] # search_travel_info,
    llm_model = ChatOpenAI(temperature=0, model="gpt-4.1-mini", use_responses_api=True)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        remaining_steps: RemainingSteps

    travel_info_agent = create_react_agent(
        model=llm_model,
        tools=tools,
        state_schema=AgentState,
        name="travel_info_agent",
        prompt="You are a helpful assistant that can search travel information and get the weather forecast. Only use the tools to find the information you need (including town names).",
    )
    await chat_loop(travel_info_agent)

if __name__ == "__main__":
    asyncio.run(main()) 