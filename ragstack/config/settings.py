"""
Configuration settings for the RAG application.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Union

# Load environment variables from .env file
load_dotenv()

# Base directories - use environment variables with defaults
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = Path(os.getenv('DATA_DIR', './data')).absolute() if os.getenv('DATA_DIR', '').startswith('./') else Path(os.getenv('DATA_DIR', './data'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', './logs')).absolute() if os.getenv('LOGS_DIR', '').startswith('./') else Path(os.getenv('LOGS_DIR', './logs'))

# Data directories - use environment variables with defaults
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR', './data/raw')).absolute() if os.getenv('RAW_DATA_DIR', '').startswith('./') else Path(os.getenv('RAW_DATA_DIR', './data/raw'))
PROCESSED_DATA_DIR = Path(os.getenv('PROCESSED_DATA_DIR', './data/processed')).absolute() if os.getenv('PROCESSED_DATA_DIR', '').startswith('./') else Path(os.getenv('PROCESSED_DATA_DIR', './data/processed'))
FAILED_DATA_DIR = Path(os.getenv('FAILED_DATA_DIR', './data/failed')).absolute() if os.getenv('FAILED_DATA_DIR', '').startswith('./') else Path(os.getenv('FAILED_DATA_DIR', './data/failed'))

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FAILED_DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Logging settings
LOG_FILE = LOGS_DIR / "ragstack.log"
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Vector database settings
VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'chroma')  # Options: "qdrant", "chroma"
VECTOR_DB_SETTINGS = {
    "qdrant": {
        "host": os.getenv('QDRANT_HOST', 'localhost'),
        "port": int(os.getenv('QDRANT_PORT', '6333')),
        "grpc_port": int(os.getenv('QDRANT_GRPC_PORT', '6334')),
        "collection_name": os.getenv('COLLECTION_NAME', 'documents'),
    },
    "chroma": {
        "host": os.getenv('CHROMA_HOST', 'http://localhost:8000'),
        "collection_name": os.getenv('COLLECTION_NAME', 'documents'),
        "persist_directory": os.getenv('CHROMA_PERSIST_DIR', str(ROOT_DIR / "chromadb-data")),
    }
}

# Embedding model settings
EMBEDDING_MODEL = {
    "name": os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2'),
    "vector_size": int(os.getenv('EMBEDDING_VECTOR_SIZE', '384')),
    "device": "cuda" if os.getenv('USE_GPU', '0') == "1" else "cpu",
    "server_host": os.getenv('EMBEDDING_SERVER_HOST', 'localhost')
}

# LLM settings for generation
LLM_SETTINGS = {
    "provider": os.getenv('LLM_PROVIDER', 'ollama'),  # Options: "openai", "anthropic", "ollama", "local"
    "host": os.getenv('LLM_HOST', 'http://localhost:11434'),  # For Ollama or local API server
    "model_name": os.getenv('LLM_MODEL_NAME', 'llama2'),  # Model name for the provider
    "api_key": os.getenv('OPENAI_API_KEY', ''),  # OpenAI API key
    "anthropic_api_key": os.getenv('ANTHROPIC_API_KEY', ''),  # Anthropic API key
    "max_tokens": int(os.getenv('LLM_MAX_TOKENS', '1024')),  # Maximum tokens to generate
    "temperature": float(os.getenv('LLM_TEMPERATURE', '0.7')),  # Temperature for generation
}

# Chunking settings
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))

# Web UI settings
WEB_HOST = os.getenv('WEB_HOST', 'localhost')
WEB_PORT = int(os.getenv('WEB_PORT', '8501'))
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
DEBUG = os.getenv('DEBUG', '0') == '1'

@dataclass
class Settings:
    """Global settings for the application."""
    # Paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    logs_dir: Path = LOGS_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    failed_data_dir: Path = FAILED_DATA_DIR
    
    # Logging
    log_file: Path = LOG_FILE
    log_level: str = LOG_LEVEL
    log_format: str = LOG_FORMAT
    
    # Vector DB
    vector_db_type: str = VECTOR_DB_TYPE
    
    @property
    def vector_db_settings(self) -> Dict:
        return VECTOR_DB_SETTINGS
    
    @property
    def embedding_model(self) -> Dict:
        return EMBEDDING_MODEL
    
    @property
    def llm_settings(self) -> Dict:
        return LLM_SETTINGS
    
    # Chunking
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    
    # Web
    web_host: str = WEB_HOST
    web_port: int = WEB_PORT
    api_host: str = API_HOST
    api_port: int = API_PORT
    debug: bool = DEBUG


# Create a global settings object
settings = Settings()