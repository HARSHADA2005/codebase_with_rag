"""
Configuration settings for the RAG application.
Centralizes all configuration parameters for easy tuning.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOAD_DIR, EMBEDDINGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Document Processing Settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks for context preservation
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt', '.csv']

# Embedding Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient model
EMBEDDING_DIMENSION = 384  # Dimension of the embedding vectors
BATCH_SIZE = 32  # Batch size for embedding generation

# Search Settings
TOP_K_RESULTS = 5  # Number of top results to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score (0-1)

# OpenAI Settings
DEFAULT_MODEL = "gpt-3.5-turbo"  # Default chat model
MAX_TOKENS = 1000  # Maximum tokens in response
TEMPERATURE = 0.7  # Creativity level (0-1)
STREAM_RESPONSE = True  # Enable streaming for better UX

# UI Theme Settings
THEME = {
    "primary_color": "#6366f1",  # Indigo
    "secondary_color": "#8b5cf6",  # Purple
    "background_color": "#0f172a",  # Dark slate
    "text_color": "#f1f5f9",  # Light slate
    "accent_color": "#10b981",  # Emerald
    "error_color": "#ef4444",  # Red
    "warning_color": "#f59e0b",  # Amber
}

# Performance Settings
ENABLE_CACHING = True
CACHE_TTL = 3600  # Cache time-to-live in seconds
MAX_CACHE_SIZE_MB = 500  # Maximum cache size

# Logging
LOG_LEVEL = "INFO"
