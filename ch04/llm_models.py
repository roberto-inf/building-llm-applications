from langchain_openai import ChatOpenAI

openai_api_key = 'YOUR_API_KEY'  # replace your key

def get_llm():
    return ChatOpenAI(openai_api_key=openai_api_key,
                 model_name="gpt-4o-mini")
