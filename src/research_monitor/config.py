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

    # Phase 4: Threshold Calibration and Evaluation
    # Provisional levels before calibration (Section 7.2 of Roadmap)
    PROVISIONAL_P_HIGH = float(os.getenv("PROVISIONAL_P_HIGH", "98.0"))
    PROVISIONAL_P_LOW = float(os.getenv("PROVISIONAL_P_LOW", "90.0"))

    # Calibrated levels (populated from dev-split calibration or environment)
    CALIBRATED_P_HIGH = (
        float(os.getenv("CALIBRATED_P_HIGH"))
        if os.getenv("CALIBRATED_P_HIGH")
        else None
    )
    CALIBRATED_P_LOW = (
        float(os.getenv("CALIBRATED_P_LOW"))
        if os.getenv("CALIBRATED_P_LOW")
        else None
    )

    # Calibration targets (Section 7.2)
    CALIBRATION_TARGET_PRECISION = float(os.getenv("CALIBRATION_TARGET_PRECISION", "0.80"))
    CALIBRATION_TARGET_MISS_RATE = float(os.getenv("CALIBRATION_TARGET_MISS_RATE", "0.10"))

    # Evaluation sampling settings (Section 8.1)
    EVALUATION_SAMPLE_SIZE = int(os.getenv("EVALUATION_SAMPLE_SIZE", "500"))
    EVALUATION_DEV_RATIO = float(os.getenv("EVALUATION_DEV_RATIO", "0.70"))
    EVALUATION_SEED = int(os.getenv("EVALUATION_SEED", "42"))

    # Access logging
    TEST_ACCESS_LOG_PATH = os.getenv("TEST_ACCESS_LOG_PATH", "experiments/results/test_access.log")

    # Phase 7: Gated LLM Summarization configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    SUMMARY_PROMPT_VERSION = os.getenv("SUMMARY_PROMPT_VERSION", "v1.0")

    @property
    def active_p_high(self) -> float:
        """Returns calibrated P_high if set, otherwise provisional P_high."""
        return self.CALIBRATED_P_HIGH if self.CALIBRATED_P_HIGH is not None else self.PROVISIONAL_P_HIGH

    @property
    def active_p_low(self) -> float:
        """Returns calibrated P_low if set, otherwise provisional P_low."""
        return self.CALIBRATED_P_LOW if self.CALIBRATED_P_LOW is not None else self.PROVISIONAL_P_LOW


config = Config()

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg2://postgres:postgres@localhost:5432/research_monitor"
)

GEMINI_API_KEY = config.GEMINI_API_KEY
LLM_MODEL = config.LLM_MODEL
LLM_TEMPERATURE = config.LLM_TEMPERATURE
SUMMARY_PROMPT_VERSION = config.SUMMARY_PROMPT_VERSION
