# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import os
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
from dotenv import load_dotenv
import random


from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
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

@tool(description="Get the weather forecast, given a town name.")
def weather_forecast(town: str) -> dict:
    """Get a mock weather forecast for a given town (Newquay, Falmouth, Truro, St Austell, Camborne, Hayle, Penzance, Land's End). Returns a WeatherForecast object with weather and temperature."""
    forecast = WeatherForecastService.get_forecast(town)
    if forecast is None:
        return {"error": f"No weather data available for '{town}'."}
    return forecast

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------
TOOLS = [search_travel_info, weather_forecast] #A

llm_model = ChatOpenAI(temperature=0, model="gpt-4.1-mini", #B
                       use_responses_api=True) #B


#A Define the tools list (in our case, only one tool)
#B Instantiate the LLM model with the gpt-4.1-mini model and the responses API

# ----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------
class AgentState(TypedDict): #A
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps #B

#A Define the agent state
#B this is a special type of state that contains the remaining steps of the agent

# ----------------------------------------------------------------------------
# Build the travel info assistant React Agent
# ----------------------------------------------------------------------------

travel_info_agent = create_react_agent(
    model=llm_model,
    tools=TOOLS,
    state_schema=AgentState,
    prompt="You are a helpful assistant that can search travel information and get the weather forecast. Only use the tools to find the information you need (including town names).",
)

# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

def chat_loop(): #A
    print("UK Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip() #B
        if user_input.lower() in {"exit", "quit"}: #C
            break
        state = {"messages": [HumanMessage(content=user_input)]} #D
        result = accommodation_booking_agent.invoke(state) #E
        response_msg = result["messages"][-1] #F
        print(f"Assistant: {response_msg.content}\n") #G

#A Define the chat loop
#B Get the user input
#C Check if the user input is "exit" or "quit" to exit the loop
#D Create the initial state with a HumanMessage containing the user input
#E Invoke the graph with the initial state
#F Get the last message from the result, which contains the final answer
#G Print the assistant's final answer, from the content of the last message


# -----------------------------------------------------------------------------
# WeatherForecastService (Mock)
# -----------------------------------------------------------------------------

class WeatherForecast(TypedDict):
    town: str
    weather: Literal["sunny", "foggy", "rainy", "windy"]
    temperature: int

class WeatherForecastService:

    _weather_options = ["sunny", "foggy", "rainy", "windy"]
    _temp_min = 18
    _temp_max = 31

    @classmethod
    def get_forecast(cls, town: str) -> Optional[WeatherForecast]: #A
        weather = random.choice(cls._weather_options)
        temperature = random.randint(cls._temp_min, cls._temp_max)
        return WeatherForecast(town=town, weather=weather, temperature=temperature)

#A Define the get_forecast method, which returns a WeatherForecast object

# -----------------------------------------------------------------------------
# HotelBookingService (MySQL-based, mock implementation)
# -----------------------------------------------------------------------------

class HotelOffer(TypedDict):
    hotel_id: int
    hotel_name: str
    town: str
    available_rooms: int
    price_per_room: float

class HotelBookingService:
    @staticmethod
    def get_offers_near_town(town: str, num_rooms: int) -> List[HotelOffer]:
        # TODO: Replace this mock with actual MySQL query logic
        # Example mock data for demonstration
        mock_offers = [
            {"hotel_id": 1, "hotel_name": "Seaview Hotel", "town": "Newquay", "available_rooms": 5, "price_per_room": 120.0},
            {"hotel_id": 2, "hotel_name": "Harbour Inn", "town": "Falmouth", "available_rooms": 2, "price_per_room": 95.0},
            {"hotel_id": 3, "hotel_name": "Cornish Retreat", "town": "St Austell", "available_rooms": 8, "price_per_room": 110.0},
            {"hotel_id": 4, "hotel_name": "Penzance Palace", "town": "Penzance", "available_rooms": 3, "price_per_room": 130.0},
            {"hotel_id": 5, "hotel_name": "The Camborne Arms", "town": "Camborne", "available_rooms": 4, "price_per_room": 105.0},
            {"hotel_id": 6, "hotel_name": "Hayle Haven", "town": "Hayle", "available_rooms": 6, "price_per_room": 99.0},
            {"hotel_id": 7, "hotel_name": "Land's End Lodge", "town": "Land's End", "available_rooms": 2, "price_per_room": 150.0},
            {"hotel_id": 8, "hotel_name": "Bude Beach Hotel", "town": "Bude", "available_rooms": 7, "price_per_room": 115.0},
            {"hotel_id": 9, "hotel_name": "Padstow Quay Inn", "town": "Padstow", "available_rooms": 5, "price_per_room": 125.0},
            {"hotel_id": 10, "hotel_name": "St Ives Bay Resort", "town": "St Ives", "available_rooms": 6, "price_per_room": 140.0},
            {"hotel_id": 11, "hotel_name": "Looe Harbour Hotel", "town": "Looe", "available_rooms": 3, "price_per_room": 108.0},
            {"hotel_id": 12, "hotel_name": "Polperro Cove Inn", "town": "Polperro", "available_rooms": 4, "price_per_room": 112.0},
            {"hotel_id": 13, "hotel_name": "Mevagissey Seaside Hotel", "town": "Mevagissey", "available_rooms": 5, "price_per_room": 118.0},
            {"hotel_id": 14, "hotel_name": "Port Isaac Retreat", "town": "Port Isaac", "available_rooms": 2, "price_per_room": 135.0},
            {"hotel_id": 15, "hotel_name": "Fowey Riverside Hotel", "town": "Fowey", "available_rooms": 3, "price_per_room": 122.0},
        ]
        # Filter for offers in the requested town with enough rooms
        offers = [offer for offer in mock_offers if offer["town"].lower() == town.lower() and offer["available_rooms"] >= num_rooms]
        return offers

@tool(description="Check hotel room availability and price for a destination in Cornwall.")
def check_hotel_availability(destination: str, num_rooms: int) -> List[Dict]:
    """Check hotel room availability and price for the requested destination and number of rooms."""
    offers = HotelBookingService.get_offers_near_town(destination, num_rooms)
    if not offers:
        return [{"error": f"No available hotels found in {destination} for {num_rooms} rooms."}]
    return offers

# -----------------------------------------------------------------------------
# BnBBookingService (Mock REST API client)
# -----------------------------------------------------------------------------

class BnBOffer(TypedDict):
    bnb_id: int
    bnb_name: str
    town: str
    available_rooms: int
    price_per_room: float

class BnBBookingService:
    @staticmethod
    def get_offers_near_town(town: str, num_rooms: int) -> List[BnBOffer]:
        # Mocked REST API response: multiple BnBs per destination
        mock_bnb_offers = [
            # Newquay
            {"bnb_id": 1, "bnb_name": "Seaside BnB", "town": "Newquay", "available_rooms": 3, "price_per_room": 80.0},
            {"bnb_id": 2, "bnb_name": "Surfside Guesthouse", "town": "Newquay", "available_rooms": 2, "price_per_room": 85.0},
            # Falmouth
            {"bnb_id": 3, "bnb_name": "Harbour View BnB", "town": "Falmouth", "available_rooms": 4, "price_per_room": 78.0},
            {"bnb_id": 4, "bnb_name": "Seafarer's Rest", "town": "Falmouth", "available_rooms": 1, "price_per_room": 90.0},
            # St Austell
            {"bnb_id": 5, "bnb_name": "Garden Gate BnB", "town": "St Austell", "available_rooms": 2, "price_per_room": 82.0},
            {"bnb_id": 6, "bnb_name": "Coastal Cottage BnB", "town": "St Austell", "available_rooms": 3, "price_per_room": 88.0},
            # Penzance
            {"bnb_id": 7, "bnb_name": "Penzance Pier BnB", "town": "Penzance", "available_rooms": 2, "price_per_room": 95.0},
            {"bnb_id": 8, "bnb_name": "Cornish Charm BnB", "town": "Penzance", "available_rooms": 3, "price_per_room": 87.0},
            # Camborne
            {"bnb_id": 9, "bnb_name": "Camborne Corner BnB", "town": "Camborne", "available_rooms": 2, "price_per_room": 75.0},
            {"bnb_id": 10, "bnb_name": "Rose Cottage BnB", "town": "Camborne", "available_rooms": 2, "price_per_room": 79.0},
            # Hayle
            {"bnb_id": 11, "bnb_name": "Hayle Haven BnB", "town": "Hayle", "available_rooms": 3, "price_per_room": 83.0},
            {"bnb_id": 12, "bnb_name": "Dune View BnB", "town": "Hayle", "available_rooms": 1, "price_per_room": 81.0},
            # Land's End
            {"bnb_id": 13, "bnb_name": "Land's End Lookout BnB", "town": "Land's End", "available_rooms": 2, "price_per_room": 100.0},
            {"bnb_id": 14, "bnb_name": "Atlantic Edge BnB", "town": "Land's End", "available_rooms": 2, "price_per_room": 105.0},
            # Bude
            {"bnb_id": 15, "bnb_name": "Bude Beach BnB", "town": "Bude", "available_rooms": 2, "price_per_room": 77.0},
            {"bnb_id": 16, "bnb_name": "Cliffside BnB", "town": "Bude", "available_rooms": 3, "price_per_room": 80.0},
            # Padstow
            {"bnb_id": 17, "bnb_name": "Padstow Harbour BnB", "town": "Padstow", "available_rooms": 2, "price_per_room": 92.0},
            {"bnb_id": 18, "bnb_name": "Fisherman's Rest BnB", "town": "Padstow", "available_rooms": 2, "price_per_room": 89.0},
            # St Ives
            {"bnb_id": 19, "bnb_name": "St Ives Bay BnB", "town": "St Ives", "available_rooms": 3, "price_per_room": 97.0},
            {"bnb_id": 20, "bnb_name": "Artists' Retreat BnB", "town": "St Ives", "available_rooms": 2, "price_per_room": 102.0},
            # Looe
            {"bnb_id": 21, "bnb_name": "Looe Riverside BnB", "town": "Looe", "available_rooms": 2, "price_per_room": 84.0},
            {"bnb_id": 22, "bnb_name": "Harbour Lights BnB", "town": "Looe", "available_rooms": 2, "price_per_room": 86.0},
            # Polperro
            {"bnb_id": 23, "bnb_name": "Polperro Cove BnB", "town": "Polperro", "available_rooms": 2, "price_per_room": 91.0},
            {"bnb_id": 24, "bnb_name": "Smuggler's Rest BnB", "town": "Polperro", "available_rooms": 2, "price_per_room": 93.0},
            # Mevagissey
            {"bnb_id": 25, "bnb_name": "Mevagissey Harbour BnB", "town": "Mevagissey", "available_rooms": 2, "price_per_room": 90.0},
            {"bnb_id": 26, "bnb_name": "Seafarer's BnB", "town": "Mevagissey", "available_rooms": 2, "price_per_room": 88.0},
            # Port Isaac
            {"bnb_id": 27, "bnb_name": "Port Isaac View BnB", "town": "Port Isaac", "available_rooms": 2, "price_per_room": 99.0},
            {"bnb_id": 28, "bnb_name": "Fisherman's Cottage BnB", "town": "Port Isaac", "available_rooms": 2, "price_per_room": 101.0},
            # Fowey
            {"bnb_id": 29, "bnb_name": "Fowey Quay BnB", "town": "Fowey", "available_rooms": 2, "price_per_room": 94.0},
            {"bnb_id": 30, "bnb_name": "Riverside Rest BnB", "town": "Fowey", "available_rooms": 2, "price_per_room": 96.0},
        ]
        offers = [offer for offer in mock_bnb_offers if offer["town"].lower() == town.lower() and offer["available_rooms"] >= num_rooms]
        return offers

@tool(description="Check BnB room availability and price for a destination in Cornwall.")
def check_bnb_availability(destination: str, num_rooms: int) -> List[Dict]:
    """Check BnB room availability and price for the requested destination and number of rooms."""
    offers = BnBBookingService.get_offers_near_town(destination, num_rooms)
    if not offers:
        return [{"error": f"No available BnBs found in {destination} for {num_rooms} rooms."}]
    return offers

BOOKING_TOOLS = [check_hotel_availability, check_bnb_availability]

# -----------------------------------------------------------------------------
# Accommodation Booking Agent
# -----------------------------------------------------------------------------
accommodation_booking_agent = create_react_agent(
    model=llm_model,
    tools=BOOKING_TOOLS,
    state_schema=AgentState,
    prompt="You are a helpful assistant that can check hotel and BnB room availability and price for a destination in Cornwall. You can use the tools to get the information you need. If the users does not specify the accommodation type, you should check both hotels and BnBs.",
)

if __name__ == "__main__":
    chat_loop() 