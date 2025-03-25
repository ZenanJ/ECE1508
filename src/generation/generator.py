import requests
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

# local model call
def llm_local_call(user_query, model="deepseek-r1:7b"):
    llm = Ollama(model=model)
    response = llm.invoke(user_query)
    return response

# API call
def llm_api_call(user_query, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    Make an API call to the Hugging Face model.
    
    Args:
        user_query (str): The user's movie recommendation request
        model (str): The model identifier to use
        
    Returns:
        dict: The JSON response from the API
    """
    HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    
    data = {"inputs": user_query}
    response = requests.post(os.getenv('API_URL'), headers=HEADERS, json=data)
    
    return response.json()


if __name__ == "__main__":
    # result = llm_api_call("just tell me the number of 1+1")
    result = llm_local_call("just tell me the number of 1+1")
    print(result)
    