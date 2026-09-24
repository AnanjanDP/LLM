import os
from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Compute absolute path to backend/app/.env and load with override=True
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env")

if os.path.exists(ENV_FILE_PATH):
    load_dotenv(ENV_FILE_PATH, override=True)


class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Intelligence Platform"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "dev-secret-key-change-in-production-min-32-characters"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./app.db"
    
    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = "groq"
    DEFAULT_LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: int = 60
    
    # Provider API Keys
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    
    # RAG Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FAISS_INDEX_PATH: str = "data/faiss.index"
    FAISS_DOCS_PATH: str = "data/documents.npy"
    RAG_TOP_K: int = 4
    DEFAULT_RETRIEVAL_MODE: str = "hybrid"  # vector, bm25, hybrid
    ENABLE_RERANKER: bool = False
    RRF_K: int = 60
    CHUNK_STRATEGY: str = "word"  # word, sentence, paragraph
    CHUNK_SIZE: int = 200
    CHUNK_OVERLAP: int = 40
    
    # Redis & Cache Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600
    
    # Prompt Versioning
    DEFAULT_PROMPT_VERSION: str = "v1.0"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: str = "60/minute"
    
    # CORS Origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


settings = Settings()
