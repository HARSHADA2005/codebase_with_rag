"""
Vector store module for semantic search.
Handles embedding generation and similarity search.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config
import utils


class VectorStore:
    """Manages document embeddings and semantic search."""
    
    def __init__(self):
        self.model = None
        self.embeddings = []
        self.chunks = []
        self.metadata = []
        self.is_initialized = False
    
    @st.cache_resource
    def _load_model(_self):
        """Load the sentence transformer model (cached)."""
        return SentenceTransformer(config.EMBEDDING_MODEL)
    
    def initialize(self):
        """Initialize the vector store and load the model."""
        if not self.is_initialized:
            with st.spinner("🔄 Loading embedding model..."):
                self.model = self._load_model()
                self.is_initialized = True
    
    def add_documents(self, documents: List[Dict], show_progress: bool = True):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of processed document dictionaries
            show_progress: Whether to show progress bar
        """
        if not self.is_initialized:
            self.initialize()
        
        all_chunks = []
        all_metadata = []
        
        # Collect all chunks and metadata
        for doc in documents:
            for i, chunk in enumerate(doc['chunks']):
                all_chunks.append(chunk)
                all_metadata.append({
                    'filename': doc['filename'],
                    'chunk_index': i,
                    'total_chunks': doc['num_chunks'],
                    'file_hash': doc['file_hash']
                })
        
        if not all_chunks:
            return
        
        # Generate embeddings in batches
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        new_embeddings = []
        batch_size = config.BATCH_SIZE
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            new_embeddings.extend(batch_embeddings)
            
            if show_progress:
                progress = min((i + batch_size) / len(all_chunks), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Embedding chunks: {i + len(batch)}/{len(all_chunks)}")
        
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        
        # Add to store
        self.embeddings.extend(new_embeddings)
        self.chunks.extend(all_chunks)
        self.metadata.extend(all_metadata)
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with chunks and metadata
        """
        if not self.embeddings:
            return []
        
        top_k = top_k or config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold and prepare results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= config.SIMILARITY_THRESHOLD:
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'score': score
                })
        
        return results
    
    def clear(self):
        """Clear all stored embeddings and chunks."""
        self.embeddings = []
        self.chunks = []
        self.metadata = []
    
    def save(self, filepath: Path):
        """
        Save the vector store to disk.
        
        Args:
            filepath: Path to save the store
        """
        data = {
            'embeddings': self.embeddings,
            'chunks': self.chunks,
            'metadata': self.metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: Path):
        """
        Load the vector store from disk.
        
        Args:
            filepath: Path to load the store from
        """
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data['embeddings']
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            
            return True
        except Exception as e:
            utils.show_error(f"Error loading vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'num_chunks': len(self.chunks),
            'num_embeddings': len(self.embeddings),
            'embedding_dimension': config.EMBEDDING_DIMENSION,
            'model': config.EMBEDDING_MODEL
        }


def format_search_results(results: List[Dict]) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Formatted string
    """
    if not results:
        return "No relevant results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        score = result['score']
        chunk = result['chunk']
        
        # Truncate chunk for display
        display_chunk = chunk[:200] + "..." if len(chunk) > 200 else chunk
        
        formatted.append(
            f"**Result {i}** (Score: {score:.3f})\n"
            f"📄 {metadata['filename']} - Chunk {metadata['chunk_index'] + 1}/{metadata['total_chunks']}\n"
            f"> {display_chunk}\n"
        )
    
    return "\n".join(formatted)
