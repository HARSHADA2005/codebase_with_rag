"""
Utility functions for the RAG application.
Provides helper functions for file handling, text processing, and validation.
"""

import hashlib
import re
import time
from pathlib import Path
from typing import List, Optional
import streamlit as st

import config


def validate_file(file) -> tuple[bool, str]:
    """
    Validate uploaded file for size and format.
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file is None:
        return False, "No file provided"
    
    # Check file extension
    file_ext = Path(file.name).suffix.lower()
    if file_ext not in config.SUPPORTED_FORMATS:
        return False, f"Unsupported format. Supported: {', '.join(config.SUPPORTED_FORMATS)}"
    
    # Check file size
    file_size_mb = file.size / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        return False, f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
    
    return True, ""


def get_file_hash(file_content: bytes) -> str:
    """
    Generate a unique hash for file content.
    Used for caching and deduplication.
    
    Args:
        file_content: File content as bytes
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(file_content).hexdigest()


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            sentence_end = max(
                text.rfind('. ', start, end),
                text.rfind('! ', start, end),
                text.rfind('? ', start, end)
            )
            if sentence_end > start:
                end = sentence_end + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def show_success(message: str):
    """Display success message with custom styling."""
    st.success(f"✅ {message}")


def show_error(message: str):
    """Display error message with custom styling."""
    st.error(f"❌ {message}")


def show_warning(message: str):
    """Display warning message with custom styling."""
    st.warning(f"⚠️ {message}")


def show_info(message: str):
    """Display info message with custom styling."""
    st.info(f"ℹ️ {message}")


class Timer:
    """Simple timer for performance monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        return self
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    def elapsed_str(self) -> str:
        """Get formatted elapsed time."""
        elapsed = self.elapsed()
        if elapsed < 1:
            return f"{elapsed*1000:.0f}ms"
        return f"{elapsed:.2f}s"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = Path(filename).name
    
    # Remove or replace unsafe characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    
    return filename


def get_cache_key(*args) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        *args: Arguments to hash
        
    Returns:
        Cache key string
    """
    key_str = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()
