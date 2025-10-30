# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto")  # "auto", "local", "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # change as available
EMBEDDING_MODEL_OPENAI = os.getenv("EMBEDDING_MODEL_OPENAI", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
