"""
Embeddings model utilities for the RAG application.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ragstack.config.settings import settings
from ragstack.utils.logging import setup_logger


logger = setup_logger("ragstack.models.embeddings")


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a query text into an embedding.
        
        Args:
            query: Query text to encode
            
        Returns:
            Embedding vector
        """
        pass


class DeepSeekEmbeddings(EmbeddingModel):
    """Embedding model using DeepSeek models."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        server_host: Optional[str] = None
    ):
        """
        Initialize DeepSeek embeddings model.
        
        Args:
            model_name: Name of the DeepSeek model to use
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum sequence length for tokenization
            server_host: Host address for remote embedding server
        """
        # Use settings if not provided
        self.model_name = model_name or settings.embedding_model["name"]
        self.device = device or settings.embedding_model["device"]
        self.max_length = max_length
        self.server_host = server_host or settings.embedding_model.get("server_host", "192.168.1.49") 
        self.use_remote = self.server_host is not None
        
        if self.use_remote:
            logger.info(f"Using remote DeepSeek model on server: {self.server_host}")
            # For remote server, we don't need to load the model
            self.api_url = f"http://{self.server_host}:8000/embed"
            
            # Check if remote server is available
            try:
                import requests
                import time
                
                logger.info(f"Testing connection to embedding server at {self.server_host}")
                start_time = time.time()
                
                try:
                    response = requests.post(
                        self.api_url,
                        json={"texts": ["Hello world"]},
                        timeout=5  # Short timeout for test
                    )
                    response.raise_for_status()
                    response_time = time.time() - start_time
                    
                    logger.info(f"Connection to embedding server successful! Response time: {response_time:.2f}s")
                    logger.info(f"Using remote embedding API at {self.api_url}")
                except requests.exceptions.ConnectionError:
                    logger.error(f"Could not connect to embedding server at {self.server_host}")
                    logger.warning("Will attempt to connect during embedding generation")
                except requests.exceptions.Timeout:
                    logger.error(f"Connection to embedding server timed out")
                    logger.warning("Embedding server may be overloaded, will retry during embedding generation")
                except Exception as e:
                    logger.error(f"Error testing embedding server: {e}")
                    logger.warning("Will attempt to connect during embedding generation")
                
            except Exception as e:
                logger.error(f"Error setting up remote embedding client: {e}")
                raise ConnectionError(f"Remote embedding setup failed: {e}")
        else:
            # Load model and tokenizer locally
            logger.info(f"Loading DeepSeek model locally: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            logger.info(f"DeepSeek model loaded on {self.device}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into embeddings."""
        import time
        
        if self.use_remote:
            # Use remote server for embeddings
            import requests
            import numpy as np
            
            try:
                logger.debug(f"Requesting embeddings for {len(texts)} texts from remote server")
                start_time = time.time()
                
                response = requests.post(
                    self.api_url,
                    json={"texts": texts},
                    timeout=60  # Increased timeout for larger batches
                )
                response.raise_for_status()
                
                embeddings = response.json()["embeddings"]
                elapsed_time = time.time() - start_time
                
                logger.debug(f"Received {len(embeddings)} embeddings from remote server in {elapsed_time:.2f}s")
                
                # Validate embeddings
                for i, embedding in enumerate(embeddings):
                    if not embedding or len(embedding) != settings.embedding_model["vector_size"]:
                        logger.warning(f"Invalid embedding at index {i}, using random fallback")
                        embeddings[i] = np.random.rand(settings.embedding_model["vector_size"]).tolist()
                
                return embeddings
                
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error with embedding server: {e}")
                logger.warning("Using random embeddings as fallback due to connection error")
                vector_size = settings.embedding_model["vector_size"]
                return [np.random.rand(vector_size).tolist() for _ in texts]
                
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout error with embedding server: {e}")
                logger.warning("Using random embeddings as fallback due to timeout")
                vector_size = settings.embedding_model["vector_size"]
                return [np.random.rand(vector_size).tolist() for _ in texts]
                
            except Exception as e:
                logger.error(f"Error getting embeddings from remote server: {e}", exc_info=True)
                logger.warning("Using random embeddings as fallback")
                vector_size = settings.embedding_model["vector_size"]
                return [np.random.rand(vector_size).tolist() for _ in texts]
        
        else:
            # Use local model
            embeddings = []
            total_start = time.time()
            
            logger.debug(f"Generating embeddings locally for {len(texts)} texts")
            
            try:
                for i, text in enumerate(texts):
                    text_start = time.time()
                    logger.debug(f"Processing text {i+1}/{len(texts)} with length {len(text)}")
                    
                    # Tokenize text
                    tokenize_start = time.time()
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=self.max_length
                    ).to(self.device)
                    tokenize_time = time.time() - tokenize_start
                    logger.debug(f"Tokenization completed in {tokenize_time:.2f}s")
                    
                    # Generate embeddings
                    inference_start = time.time()
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                    inference_time = time.time() - inference_start
                    logger.debug(f"Model inference completed in {inference_time:.2f}s")
                    
                    # Extract embedding
                    embedding_start = time.time()
                    embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy().tolist()[0]
                    embedding_time = time.time() - embedding_start
                    logger.debug(f"Embedding extraction completed in {embedding_time:.2f}s")
                    
                    embeddings.append(embedding)
                    
                    text_time = time.time() - text_start
                    logger.debug(f"Text {i+1} processed in {text_time:.2f}s, embedding size: {len(embedding)}")
                
                total_time = time.time() - total_start
                logger.info(f"Generated {len(embeddings)} embeddings locally in {total_time:.2f}s")
                
                return embeddings
                
            except Exception as e:
                logger.error(f"Error generating embeddings locally: {e}", exc_info=True)
                logger.warning("Using random embeddings as fallback")
                vector_size = settings.embedding_model["vector_size"]
                return [np.random.rand(vector_size).tolist() for _ in texts]
    
    def encode_query(self, query: str) -> List[float]:
        """Encode a query text into an embedding."""
        return self.encode([query])[0]


class MockEmbeddings(EmbeddingModel):
    """Mock embedding model for testing."""
    
    def __init__(self, vector_size: int = 4096):
        """Initialize mock embeddings model."""
        self.vector_size = vector_size
        logger.warning("Using mock embeddings model - no actual semantic matching will occur")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for a list of texts."""
        return [np.random.randn(self.vector_size).tolist() for _ in texts]
    
    def encode_query(self, query: str) -> List[float]:
        """Generate a random embedding for a query."""
        return np.random.randn(self.vector_size).tolist()


def get_embeddings_model() -> EmbeddingModel:
    """
    Get the embeddings model based on configuration.
    
    Returns:
        Configured embeddings model
    """
    model_name = settings.embedding_model["name"]
    
    try:
        # Currently only support DeepSeek model
        if "deepseek" in model_name.lower():
            return DeepSeekEmbeddings()
        else:
            # Default to DeepSeek
            logger.warning(f"Unknown embedding model: {model_name}, using DeepSeek")
            return DeepSeekEmbeddings()
    except Exception as e:
        # If model loading fails, use mock embeddings
        logger.error(f"Failed to load embedding model: {e}")
        logger.warning("Using mock embeddings as fallback")
        return MockEmbeddings(vector_size=settings.embedding_model["vector_size"])