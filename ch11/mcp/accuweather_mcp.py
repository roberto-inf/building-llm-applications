import os
import json
from typing import Dict
from fastmcp import FastMCP
from dotenv import load_dotenv
from aiohttp import ClientSession

#--------------------------------
# ADAPTED FROM: https://github.com/adhikasp/mcp-weather
#--------------------------------

load_dotenv() #A

mcp = FastMCP("mcp-accuweather") #B

@mcp.tool(description="Get weather conditions for a location.") #C
async def get_weather_conditions(location: str) -> Dict:
    """Get weather conditions for a location."""
    api_key = os.getenv("ACCUWEATHER_API_KEY") #D
    base_url = "http://dataservice.accuweather.com"

    async with ClientSession() as session:
        location_search_url = f"{base_url}/locations/v1/cities/search" #E
        params = { #F
            "apikey": api_key,
            "q": location,
        }
        async with session.get(location_search_url, params=params) as response:
            locations = await response.json() #G
            if response.status != 200:
                raise Exception(f"Error fetching location data: {response.status}, {locations}")
            if not locations or len(locations) == 0:
                raise Exception("Location not found")
        location_key = locations[0]["Key"] #H

        # Get current conditions 
        current_conditions_url = f"{base_url}/currentconditions/v1/{location_key}" #I
        params = { #J
            "apikey": api_key,
            "details": "true"
        }
        async with session.get(current_conditions_url, params=params) as response:
            current_conditions = await response.json() #K
            
        # Format current conditions
        if current_conditions and len(current_conditions) > 0:
            current = current_conditions[0] #L
            current_data = {
                "temperature": {
                    "value": current["Temperature"]["Metric"]["Value"],
                    "unit": current["Temperature"]["Metric"]["Unit"]
                },
                "weather_text": current["WeatherText"],
                "relative_humidity": current.get("RelativeHumidity"),
                "precipitation": current.get("HasPrecipitation", False),
                "observation_time": current["LocalObservationDateTime"]
            }
        else:
            current_data = "No current conditions available" #M

        return { #N
            "location": locations[0]["LocalizedName"], 
            "location_key": location_key, 
            "country": locations[0]["Country"]["LocalizedName"], 
            "current_conditions": current_data,
        }

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", #O
            port=8020, path="/accu-mcp-server")
    
#A - Load environment variables
#B - Initialize FastMCP
#C - Define MCP tool
#D - Get AccuWeather API key
#E - Base URL
#F - Parameters for location search
#G - Get locations
#H - Get location key
#I - Current conditions URL
#J - Current conditions parameters
#K - Get current conditions
#L - Format current conditions
#M - No current conditions available
#N - Return structured content  
#O - Run MCP server and expose it on port 8020 as a Streamable HTTP server with a MCP endpoint at /accu-mcp-server