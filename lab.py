
# Important!!!
#
# <---- Set your 'OPENAI_API_KEY' as a secret over there with the "key" icon
#
#
import time
from litellm import completion
from typing import List, Dict

def generate_response(messages: List[Dict]) -> str:
    """Call local DeepSeek LLM via LiteLLM"""
    response = completion(
        # For Ollama, the format is 'ollama/model_name'
        # If using LM Studio, use 'openai/model_name' and set base_url
        model="ollama/deepseek-r1:14b",
        messages=messages,
        max_tokens=1024,
        api_base="http://localhost:11434" # Default port for Ollama
    )
    return response.choices[0].message.content

what_to_help_with = input("What do you need help with? ")

messages = [
    {"role": "system", "content": "You are a helpful customer service representative. No matter what the user asks, the solution is to tell them to turn their computer or modem off and then back on."},
    {"role": "user", "content": what_to_help_with}
]
startTime = time.perf_counter()
response = generate_response(messages)
print(response)

# Second query without including the previous response
messages = [
    {"role": "user", "content": "Update the function to include documentation."}
]

endTime = time.perf_counter()
print(f"response time : {endTime - startTime}")
