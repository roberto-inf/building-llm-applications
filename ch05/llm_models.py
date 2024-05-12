from langchain_openai import ChatOpenAI

openai_api_key = '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'

def get_llm():
    return ChatOpenAI(openai_api_key=openai_api_key,
                 model_name="gpt-3.5-turbo")
