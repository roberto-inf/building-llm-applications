from llm_models import get_llm
from utilities import to_obj
from prompts import (
    ASSISTANT_SELECTION_PROMPT_TEMPLATE, 
)
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

assistant_instructions_chain = (
    {'user_question': RunnablePassthrough()} 
    | ASSISTANT_SELECTION_PROMPT_TEMPLATE | get_llm() | StrOutputParser() | to_obj
)