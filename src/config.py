import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force HuggingFace to use your D: drive cache
os.environ["HF_HOME"] = os.getenv("HF_HOME", r"D:\VoxSync_Cache\huggingface")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")