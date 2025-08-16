# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import os
import uuid
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
from dotenv import load_dotenv
import random
from enum import Enum
from pydantic import BaseModel, Field


from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


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
    """Get a mock weather forecast for a given town. Returns a WeatherForecast object with weather and temperature."""
    forecast = WeatherForecastService.get_forecast(town)
    if forecast is None:
        return {"error": f"No weather data available for '{town}'."}
    return forecast

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------
TOOLS = [search_travel_info, weather_forecast] #A

#A Define the tools list (in our case, only one tool)


llm_model = ChatOpenAI(model="gpt-5", #A
                       use_responses_api=True, #B                      
                       use_previous_response_id=True) #C

#A Instantiate the LLM model with the gpt-5 model
#B Use the Responses API
#C Use the previous response ID to continue the conversation

# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------
class AgentState(TypedDict): #A
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps #B

#A Define the agent state
#B this is a special type of state that contains the remaining steps of the agent

# -----------------------------------------------------------------------------
# AgentType Enum and Structured Output Model
# -----------------------------------------------------------------------------
class AgentType(str, Enum):
    travel_info_agent = "travel_info_agent"
    accommodation_booking_agent = "accommodation_booking_agent"

class AgentTypeOutput(BaseModel): 
    agent: AgentType = Field(..., description="Which agent should handle the query?")

# Structured LLM for routing
llm_router = llm_model.with_structured_output(AgentTypeOutput)

# -----------------------------------------------------------------------------
# Guardrail: pre_model_hook to allow only travel-related questions
# -----------------------------------------------------------------------------
class GuardrailDecision(BaseModel): #A
    is_travel: bool = Field(
        ...,
        description=(
            "True if the user question is about travel information: destinations, attractions, "
            "lodging (hotels/BnBs), prices, availability, or weather in Cornwall/England."
        ),
    )
    reason: str = Field(..., description="Brief justification for the decision.")

GUARDRAIL_SYSTEM_PROMPT = ( #B
    "You are a strict classifier. Given the user's last message, respond with whether it is "
    "travel-related. Travel-related queries include destinations, attractions, lodging (hotels/BnBs), "
    "room availability, prices, or weather in Cornwall/England."
)

REFUSAL_INSTRUCTION = ( #C
    "You can only help with travel-related questions (destinations, attractions, lodging, prices, "
    "availability, or weather in Cornwall/England). The user's request is not travel-related. "
    "Politely refuse and briefly explain what topics you can help with."
)


llm_guardrail = llm_model.with_structured_output(GuardrailDecision) #D
#A Define the GuardrailDecision model
#B Define the GUARDRAIL_SYSTEM_PROMPT which constrains the model to only answer travel-related questions
#C Define the REFUSAL_INSTRUCTION which is used to politely refuse to answer non-travel-related questions
#D Use the same base model with structured output for fast, lightweight classification

AGENT_GUARDRAIL_SYSTEM_PROMPT = ( 
    "You are a strict classifier. Given the user's last message, respond with whether it is "
    "travel-related. Travel-related queries include destinations, attractions, lodging (hotels/BnBs), "
    "room availability, prices, or weather in Cornwall/England."
    "Only accept travel-related questions covering Cornwall (England) and reject any questions"
    "from other areas in England and from other countries"
)

AGENT_REFUSAL_INSTRUCTION = ( 
    "You can only help with travel-related questions (destinations, attractions, lodging, prices, "
    "availability, or weather in Cornwall/England). The user's request is not travel-related. "
    "Or it might be a travel related question but not focusing on Cornwall (England). "
    "Politely refuse and briefly explain what topics you can help with."
)

def pre_model_guardrail(state: dict):
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if not isinstance(last_msg, HumanMessage): #A
        return {}

    user_input = last_msg.content
    classifier_messages = [ #B
        SystemMessage(content=AGENT_GUARDRAIL_SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]
    decision = llm_guardrail.invoke(classifier_messages)

    if decision.is_travel: #C
        # Allow normal flow; do not modify inputs
        return {}

    # Inject a refusal instruction ahead of the original messages so the model politely declines
    return {"llm_input_messages": [SystemMessage(content=AGENT_REFUSAL_INSTRUCTION), *messages]} #D

#A Check if the last message is a HumanMessage (which is the user input)
#B Create the classifier messages, including the system prompt and the user input
#C Check if the decision is travel-related. If so, allow normal flow; do not modify inputs
#D If the decision is not travel-related, inject a refusal instruction ahead of the original messages so the model politely declines    

# -----------------------------------------------------------------------------
# Router Agent System Prompt Constant
# -----------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = (
    "You are a router. Given the following user message, decide if it is a travel information question (about destinations, attractions, or general travel info) "
    "or an accommodation booking question (about hotels, BnBs, room availability, or prices).\n"
    "If it is a travel information question, respond with 'travel_info_agent'.\n"
    "If it is an accommodation booking question, respond with 'accommodation_booking_agent'."
)

# -----------------------------------------------------------------------------
# Router Agent Node for LangGraph (with structured output)
# -----------------------------------------------------------------------------
def router_agent_node(state: AgentState) -> Command[AgentType]:
    """Router node: decides which agent should handle the user query."""
    messages = state["messages"] 
    last_msg = messages[-1] if messages else None 
    if isinstance(last_msg, HumanMessage):
        user_input = last_msg.content 

        # Guardrail classification at routing time
        classifier_messages = [
            SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT), #A
            HumanMessage(content=user_input),
        ]
        decision = llm_guardrail.invoke(classifier_messages) #B
        if not decision.is_travel: #C
            # Return refusal directly as an AI message and shortcut to END via a dedicated node
            refusal_text = ( #D
                "Sorry, I can only help with travel-related questions (destinations, attractions, "
                "lodging, prices, availability, or weather in Cornwall/England). "
                "Please rephrase your request to be travel-related."
            )
            return Command( #E
                update={"messages": [AIMessage(content=refusal_text)]},
                goto="guardrail_refusal",
            ) 

        router_messages = [ 
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ]
        router_response = llm_router.invoke(router_messages) 
        agent_name = router_response.agent.value 
        return Command(update=state, goto=agent_name) 
    
    return Command(update=state, goto=AgentType.travel_info_agent) 

#A Define the guardrail decision prompt
#B Invoke the guardrail model, which returns a GuardrailDecision object
#C Check if the decision is not travel-related
#D Define the refusal text
#E Return the command to set a refusal message in the state and go to the guardrail refusal node

# -----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# -----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Build the travel info assistant React Agent
# ----------------------------------------------------------------------------

travel_info_agent = create_react_agent(
    model=llm_model,
    tools=TOOLS,
    state_schema=AgentState,
    prompt="You are a helpful assistant that can search travel information and get the weather forecast. Only use the tools to find the information you need (including town names).",
    pre_model_hook=pre_model_guardrail, #A
)
#A Guardrail to check if the user input is travel-related and focusing on Cornwall (England)



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
# SQLDatabaseToolkit for Hotel Booking (SQLite)
# -----------------------------------------------------------------------------
hotel_db = SQLDatabase.from_uri("sqlite:///hotel_db/cornwall_hotels.db")
hotel_db_toolkit = SQLDatabaseToolkit(db=hotel_db, llm=llm_model)
hotel_db_toolkit_tools = hotel_db_toolkit.get_tools()

# -----------------------------------------------------------------------------
# BnBBookingService (Mock REST API client)
# -----------------------------------------------------------------------------

class BnBOffer(TypedDict): #A
    bnb_id: int
    bnb_name: str
    town: str
    available_rooms: int
    price_per_room: float

class BnBBookingService: #B
    @staticmethod
    def get_offers_near_town(town: str, num_rooms: int) -> List[BnBOffer]: #C
        # Mocked REST API response: multiple BnBs per destination
        mock_bnb_offers = [ #D
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
    
#A Define the return type of the BnB availability tool
#B Define the BnB availability tool
#C Call the BnB booking service to get the offers
#D Mocked BnB offers

# -----------------------------------------------------------------------------
# BnB Availability Tool
# -----------------------------------------------------------------------------

@tool(description="Check BnB room availability and price for a destination in Cornwall.") #A
def check_bnb_availability(destination: str, num_rooms: int) -> List[Dict]: #B
    """Check BnB room availability and price for the requested destination and number of rooms."""
    offers = BnBBookingService.get_offers_near_town(destination, num_rooms)
    if not offers:
        return [{"error": f"No available BnBs found in {destination} for {num_rooms} rooms."}]
    return offers


#A Define the BnB availability tool
#B Define the input and return type of the BnB availability tool

# -----------------------------------------------------------------------------
# Accommodation Booking Agent
# -----------------------------------------------------------------------------
BOOKING_TOOLS = hotel_db_toolkit_tools + [check_bnb_availability] 

accommodation_booking_agent = create_react_agent( #A
    model=llm_model,
    tools=BOOKING_TOOLS,
    state_schema=AgentState,
    prompt="You are a helpful assistant that can check hotel and BnB room availability and price for a destination in Cornwall. You can use the tools to get the information you need. If the users does not specify the accommodation type, you should check both hotels and BnBs.",
    pre_model_hook=pre_model_guardrail,
)

#A Guardrail to check if the user input is travel-related and focusing on Cornwall (England)

# -----------------------------------------------------------------------------
# Build the LangGraph graph with router, travel_info_agent, and accommodation_booking_agent
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Guardrail refusal node (no-op, used to shortcut to END)
# -----------------------------------------------------------------------------
def guardrail_refusal_node(state: AgentState): #A
    return {}

graph = StateGraph(AgentState) 
graph.add_node("router_agent", router_agent_node) 
graph.add_node("travel_info_agent", travel_info_agent) 
graph.add_node("accommodation_booking_agent", accommodation_booking_agent) 
graph.add_node("guardrail_refusal", guardrail_refusal_node) #B

graph.add_edge("travel_info_agent", END) 
graph.add_edge("accommodation_booking_agent", END) 
graph.add_edge("guardrail_refusal", END) #C

graph.set_entry_point("router_agent") 

checkpointer = InMemorySaver() 
travel_assistant = graph.compile(checkpointer=checkpointer) 

#A Define the guardrail refusal node, which is a no-op node that is used to shortcut to END 
#B Add the guardrail refusal node
#C Add the edge from the guardrail refusal node to the end


# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

def chat_loop(): #A
    thread_id=uuid.uuid1() #B
    print(f'Thread ID: {thread_id}') 
    config={"configurable": {"thread_id": thread_id}} #B

    print("UK Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip() #C
        if user_input.lower() in {"exit", "quit"}: #D
            break
        state = {"messages": [HumanMessage(content=user_input)]} #E
        result = travel_assistant.invoke(state, config=config) #F
        response_msg = result["messages"][-1] #G
        print(f"Assistant: {response_msg.content}\n") #H


#A Define the chat loop
#B Create a unique thread id
#C Check if the user input is "exit" or "quit" to exit the loop
#D Create the initial state with a HumanMessage containing the user input
#E Set the state with the HumanMessage
#F Invoke the graph with the state and the config
#G Get the last message from the result, which contains the final answer
#H Print the assistant's final answer, from the content of the last message

if __name__ == "__main__":
    chat_loop() 