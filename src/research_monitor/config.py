import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    """Project Configuration"""
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SEED_INTEREST = os.getenv(
        "SEED_INTEREST", 
        "We are interested in AI agents, autonomous systems, reinforcement learning, large language models for reasoning, and tool use. We want to discover papers that discuss agentic workflows, multi-agent systems, and planning."
    )

config = Config()
