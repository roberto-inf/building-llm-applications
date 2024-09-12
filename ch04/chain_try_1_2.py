from chain_1_2 import assistant_instructions_chain

# test chain invocation
question = 'What can I see and do in the Spanish town of Astorga?'

assistant_instructions_dict = assistant_instructions_chain.invoke(question)
print(assistant_instructions_dict)

# Result:
# {'assistant_type': 'Tour guide assistant', 'assistant_instructions': 'You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights.', 'user_question': 'What can I see and do in the Spanish town Astorga'}