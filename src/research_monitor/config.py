import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    """Project Configuration"""
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Seed interest paragraph used for cosine similarity ranking (Phase 1+)
    SEED_INTEREST = os.getenv(
        "SEED_INTEREST",
        "We are interested in AI agents, autonomous systems, reinforcement learning, large language models for reasoning, and tool use. We want to discover papers that discuss agentic workflows, multi-agent systems, and planning.",
    )

    # Phase 2: arXiv corpus configuration
    # Date range selected to capture the LLM/agent research boom.
    # Override via .env if a different window is needed.
    CORPUS_START_DATE = os.getenv("CORPUS_START_DATE", "2020-01-01")
    CORPUS_END_DATE = os.getenv("CORPUS_END_DATE", "2024-12-31")

    # arXiv categories to include (comma-separated in .env)
    CORPUS_CATEGORIES = os.getenv("CORPUS_CATEGORIES", "cs.CL,cs.LG,cs.AI")

    # Deterministic reference sample configuration
    REFERENCE_SAMPLE_SIZE = int(os.getenv("REFERENCE_SAMPLE_SIZE", "5000"))
    REFERENCE_SAMPLE_SEED = int(os.getenv("REFERENCE_SAMPLE_SEED", "42"))

    # Sentence Transformer model (Phase 1+)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


config = Config()

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg2://postgres:postgres@localhost:5432/research_monitor"
)
