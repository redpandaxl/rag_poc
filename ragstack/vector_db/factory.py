"""
Factory for creating vector database instances.
"""
from typing import Dict, Optional, Type

from ragstack.config.settings import settings
from ragstack.utils.logging import setup_logger
from ragstack.vector_db.base import VectorDB
from ragstack.vector_db.chroma import ChromaDB
from ragstack.vector_db.qdrant import QdrantDB


logger = setup_logger("ragstack.vector_db.factory")


# Registry of supported vector databases
DB_REGISTRY: Dict[str, Type[VectorDB]] = {
    "qdrant": QdrantDB,
    "chroma": ChromaDB,
}


def get_vector_db(db_type: Optional[str] = None) -> VectorDB:
    """
    Factory function to create a vector database instance.
    
    Args:
        db_type: Type of vector database to create
        
    Returns:
        Initialized vector database instance
        
    Raises:
        ValueError: If the database type is unsupported
    """
    # Use default database type from settings if not provided
    if db_type is None:
        db_type = settings.vector_db_type
    
    # Validate database type
    if db_type not in DB_REGISTRY:
        supported_dbs = ", ".join(DB_REGISTRY.keys())
        error_msg = f"Unsupported vector database type: {db_type}. Supported types: {supported_dbs}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # First try the requested database
    try:
        db_class = DB_REGISTRY[db_type]
        db_instance = db_class()
        db_instance.initialize()
        
        # Test connection if it's Qdrant
        if db_type == "qdrant" and db_instance.client:
            try:
                # Try a simple operation to verify connection
                db_instance.client.get_collections()
                logger.info(f"Successfully connected to Qdrant at {db_instance.host}")
            except Exception as e:
                logger.error(f"Qdrant connection test failed: {e}")
                logger.warning("Falling back to ChromaDB")
                # If test fails, fall back to ChromaDB
                return get_vector_db("chroma")
        
        logger.info(f"Created vector database instance of type: {db_type}")
        return db_instance
        
    except Exception as e:
        logger.error(f"Failed to initialize {db_type}: {e}")
        
        # If the requested DB isn't available and it's not already ChromaDB, fall back to ChromaDB
        if db_type != "chroma":
            logger.warning(f"Falling back to ChromaDB due to {db_type} initialization failure")
            return get_vector_db("chroma")
        else:
            # If ChromaDB also fails, re-raise the exception
            raise