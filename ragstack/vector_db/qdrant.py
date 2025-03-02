"""
Qdrant vector database implementation.
"""
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from ragstack.config.settings import settings
from ragstack.utils.logging import setup_logger
from ragstack.vector_db.base import VectorDB


logger = setup_logger("ragstack.vector_db.qdrant")


class QdrantDB(VectorDB):
    """Qdrant vector database implementation."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None, 
        grpc_port: Optional[int] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize QdrantDB client.
        
        Args:
            host: Qdrant host
            port: Qdrant HTTP port
            grpc_port: Qdrant gRPC port
            collection_name: Default collection name
        """
        # Use settings if not provided
        qdrant_settings = settings.vector_db_settings["qdrant"]
        self.host = host or qdrant_settings["host"]
        self.port = port or qdrant_settings["port"]
        self.grpc_port = grpc_port or qdrant_settings["grpc_port"]
        self.collection_name = collection_name or qdrant_settings["collection_name"]
        self.client = None
    
    def initialize(self) -> None:
        """Initialize the connection to Qdrant."""
        try:
            self.client = QdrantClient(
                host=self.host, 
                port=self.port, 
                grpc_port=self.grpc_port,
                timeout=5  # Add a reasonable timeout
            )
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            logger.warning("Falling back to local ChromaDB")
            # Don't raise error, let the factory handle fallback
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        if not self.client:
            return False
            
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False
    
    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int = 4096,
        distance: Distance = Distance.COSINE
    ) -> None:
        """Create a new collection."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
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
            # Create points list
            points = [
                models.PointStruct(
                    id=id,
                    vector=vector,
                    payload={
                        "content": document,
                        "metadata": meta
                    }
                )
                for id, vector, document, meta in zip(ids, vectors, documents, metadata)
            ]
            
            # Insert points
            self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            
            logger.info(f"Inserted {len(points)} points into collection '{collection_name}'")
            return ids
        
        except Exception as e:
            logger.error(f"Failed to insert points into collection '{collection_name}': {e}")
            raise
    
    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection."""
        try:
            search_params = {}
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
                
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                **search_params
            )
            
            # Format results
            formatted_results = []
            for res in results:
                formatted_results.append({
                    "id": res.id,
                    "content": res.payload.get("content", ""),
                    "metadata": res.payload.get("metadata", {}),
                    "score": res.score
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Failed to search in collection '{collection_name}': {e}")
            raise
    
    def delete(self, collection_name: str, ids: List[str]) -> None:
        """Delete points from the collection."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                ),
                wait=True
            )
            logger.info(f"Deleted {len(ids)} points from collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete points from collection '{collection_name}': {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        try:
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            collection_info = self.client.get_collection(collection_name)
            # Structure of the response changed in newer Qdrant versions
            return {
                "name": collection_name,  # Use the parameter since it's not in the response
                "vectors_count": getattr(collection_info, 'vectors_count', 0),
                "points_count": getattr(collection_info, 'points_count', 0),
                "status": getattr(collection_info, 'status', 'unknown'),
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            # Return basic info to avoid breaking the UI
            return {
                "name": collection_name,
                "status": "unknown",
                "error": str(e)
            }