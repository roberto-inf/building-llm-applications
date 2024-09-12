from langchain_openai import ChatOpenAI

openai_api_key = 'sk-proj-p1yN9o3BosniB9uwcgZtT3BlbkFJZ6uu58xjrBHRyuwo9jz2'

def get_llm():
    return ChatOpenAI(openai_api_key=openai_api_key,
                 model_name="gpt-4o-mini")
