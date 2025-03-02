"""
Base class for vector database interfaces.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np


class VectorDB(ABC):
    """Abstract base class for vector database interfaces."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the connection to the vector database."""
        pass
    
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a new collection."""
        pass
    
    @abstractmethod
    def insert(
        self, 
        collection_name: str, 
        vectors: List[List[float]], 
        documents: List[str], 
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Insert vectors and documents into the collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors (embeddings)
            documents: List of document content strings
            metadata: List of metadata dictionaries
            ids: Optional list of IDs for the points
            
        Returns:
            List of inserted IDs
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Vector to search for
            limit: Maximum number of results
            
        Returns:
            List of results with document content, metadata, and similarity score
        """
        pass
    
    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]) -> None:
        """
        Delete points from the collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
        """
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        pass