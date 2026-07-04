"""Embedding model wrapper using Chroma's default MiniLM (free, local, lightweight)."""

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

def get_embedding_function():
    """Get the default embedding function (all-MiniLM-L6-v2)."""
    return DefaultEmbeddingFunction()
