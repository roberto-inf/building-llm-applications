from utilities import to_obj
from chain_4_1 import search_result_text_and_summary_chain

# test chain invocation
result_url_str = '{"result_url": "https://citiesandattractions.com/spain/astorga-spain-uncovering-the-jewels-of-a-hidden-spanish-gem/", "search_query": "Astorga Spain attractions", "user_question": "What can I see and do in the Spanish town of Astorga?"}'
result_url_dict = to_obj(result_url_str)

search_text_summary = search_result_text_and_summary_chain.invoke(result_url_dict)
print(search_text_summary)

# Result:
# {'summary': 'Source Url: https://citiesandattractions.com/spain/astorga-spain-uncovering-the-jewels-of-a-hidden-spanish-gem/\nSummary: \nAstorga, Spain has several attractions including the Episcopal Palace, Cathedral of Santa Maria de Astorga, Roman Walls and Museum, Chocolate Factory Museum, Palace of Gaudi, Sierra de los Ancares, and wineries. The town is known for its history, architecture, cuisine, and natural beauty. It offers unique culinary experiences, such as the traditional dish "Cocido Maragato," and is home to various hidden gems waiting to be discovered. ', 'user_question': 'What can I see and do in the Spanish town Astorga?'}