"""Настройки из переменных окружения (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_CHUNK = 50  # фрагменты короче отбрасываются

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "demo"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
CACHE_DB = Path(os.getenv("CACHE_DB", ROOT / "cache.db"))
COLLECTION = "documents"


def require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Не задан OPENAI_API_KEY: скопируйте .env.example в .env и впишите ключ.")
    return key
