"""
ChromaDB vector database implementation.
"""
import uuid
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb import HttpClient

from ragstack.config.settings import settings
from ragstack.utils.logging import setup_logger
from ragstack.vector_db.base import VectorDB


logger = setup_logger("ragstack.vector_db.chroma")


class ChromaDB(VectorDB):
    """ChromaDB vector database implementation."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize ChromaDB client.
        
        Args:
            host: ChromaDB host (for HTTP client)
            collection_name: Default collection name
            persist_directory: Local directory for persistent storage
        """
        # Use settings if not provided
        chroma_settings = settings.vector_db_settings["chroma"]
        self.host = host or chroma_settings.get("host")
        self.persist_directory = persist_directory or chroma_settings.get("persist_directory")
        self.collection_name = collection_name or chroma_settings["collection_name"]
        self.client = None
    
    def initialize(self) -> None:
        """Initialize the connection to ChromaDB."""
        try:
            if self.persist_directory:
                # Use persistent client with local storage
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                logger.info(f"Connected to local ChromaDB at {self.persist_directory}")
            else:
                # Use HTTP client
                self.client = chromadb.HttpClient(host=self.host.replace("http://", "").split(":")[0], 
                                                port=int(self.host.split(":")[-1]))
                logger.info(f"Connected to ChromaDB at {self.host}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.list_collections()
            return any(c.name == collection_name for c in collections)
        except Exception:
            return False
    
    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int = 4096
    ) -> None:
        """Create a new collection."""
        try:
            self.client.create_collection(
                name=collection_name,
                metadata={"vector_size": vector_size}
            )
            logger.info(f"Collection '{collection_name}' created with vector size {vector_size}")
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise
    
    def insert(
        self, 
        collection_name: str, 
        vectors: List[List[float]], 
        documents: List[str], 
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Insert vectors and documents into the collection."""
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        try:
            # Get the collection
            collection = self.client.get_collection(name=collection_name)
            
            # Add documents
            collection.add(
                embeddings=vectors,
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            
            logger.info(f"Inserted {len(ids)} documents into collection '{collection_name}'")
            return ids
        
        except Exception as e:
            logger.error(f"Failed to insert documents into collection '{collection_name}': {e}")
            raise
    
    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection."""
        try:
            # Get the collection
            collection = self.client.get_collection(name=collection_name)
            
            # Search for similar vectors
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i, (doc_id, doc, meta, distance) in enumerate(zip(
                results.get("ids", [[]])[0],
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0]
            )):
                # Convert distance to score (Chroma uses L2 distance by default)
                # Lower distance = higher score, so we invert it
                score = 1.0 / (1.0 + distance)
                
                formatted_results.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "score": score
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Failed to search in collection '{collection_name}': {e}")
            raise
    
    def delete(self, collection_name: str, ids: List[str]) -> None:
        """Delete documents from the collection."""
        try:
            # Get the collection
            collection = self.client.get_collection(name=collection_name)
            
            # Delete documents
            collection.delete(ids=ids)
            
            logger.info(f"Deleted {len(ids)} documents from collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete documents from collection '{collection_name}': {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            collection = self.client.get_collection(name=collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            raise