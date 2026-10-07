import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    """Project Configuration"""
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    # Add other configuration variables here as needed in the future

config = Config()
