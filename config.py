"""
AMARA - Configuration Module
Loads environment variables and sets up LLM clients.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Settings ────────────────────────────────────────────────────────────
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL  = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

# ── RAG Settings ─────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/faiss_index")
MAX_PAPERS        = int(os.getenv("MAX_PAPERS", 10))
MAX_CHUNKS        = int(os.getenv("MAX_CHUNKS", 5))

# ── API Keys ─────────────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


def get_llm(temperature: float = 0.3):
    """Return a LangChain LLM based on the configured provider."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=temperature,
        )
