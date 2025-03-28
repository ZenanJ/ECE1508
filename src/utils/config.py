import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API keys and configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = os.getenv("API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME")

# Default values if environment variables are not set
if not HUGGINGFACE_API_KEY:
    HUGGINGFACE_API_KEY = ""  # Set this in your .env file
if not API_URL:
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = ""  # Set this in your .env file
if not LOCAL_MODEL_NAME:
    LOCAL_MODEL_NAME = "deepseek-r1:7b"