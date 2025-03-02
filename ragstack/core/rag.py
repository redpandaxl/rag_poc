"""
Core RAG functionality for the application.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import requests
import json

from ragstack.config.settings import settings
from ragstack.models.embeddings import get_embeddings_model
from ragstack.utils.logging import setup_logger
from ragstack.vector_db.factory import get_vector_db


logger = setup_logger("ragstack.core.rag")


class RAGEngine:
    """Core RAG engine for retrieving and generating responses."""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_db_type: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Initialize RAG engine.
        
        Args:
            collection_name: Name of the collection to search
            vector_db_type: Type of vector database to use
            top_k: Number of top results to retrieve
        """
        self.vector_db_type = vector_db_type or settings.vector_db_type
        db_settings = settings.vector_db_settings[self.vector_db_type]
        self.collection_name = collection_name or db_settings["collection_name"]
        self.top_k = top_k
        
        # Components will be initialized on first use
        self._embedding_model = None
        self._vector_db = None
    
    @property
    def embedding_model(self):
        """Get the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = get_embeddings_model()
        return self._embedding_model
    
    @property
    def vector_db(self):
        """Get the vector database client."""
        if self._vector_db is None:
            self._vector_db = get_vector_db(self.vector_db_type)
        return self._vector_db
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: User query to search for
            top_k: Number of top results to retrieve
            filter_criteria: Optional criteria to filter results
            
        Returns:
            List of relevant documents with metadata and scores
        """
        # Use instance default if not provided
        top_k = top_k or self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode_query(query)
            
            # Search in vector database
            results = self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def get_retrieval_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get retrieval context as a single string for RAG.
        
        Args:
            query: User query
            top_k: Number of top results to retrieve
            
        Returns:
            Concatenated context from relevant documents
        """
        results = self.search(query, top_k)
        
        if not results:
            return ""
        
        # Format results into a single context string
        context_parts = []
        
        for i, result in enumerate(results):
            # Extract information
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            title = metadata.get("title", "Untitled")
            
            # Format context entry
            context_parts.append(
                f"[Document {i+1}] {title}\n"
                f"Source: {source}\n"
                f"{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_response(
        self, 
        query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a response using the LLM with retrieved context.
        
        Args:
            query: User query
            chat_history: Optional chat history for context
            temperature: Temperature for LLM generation (higher = more creative)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated response, sources used)
        """
        # Get context from vector database
        context = self.get_retrieval_context(query)
        
        # Get sources from results
        sources = []
        results = self.search(query)
        
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            title = metadata.get("title", "Untitled")
            sources.append({
                "title": title,
                "source": source,
                "index": i+1
            })
        
        if not context:
            return "I couldn't find any relevant information in the knowledge base for your query.", []
        
        # Format prompt for the LLM with context
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from a knowledge base.
Your goal is to provide accurate, helpful, and concise responses using ONLY the information in the CONTEXT.
If the answer cannot be found in the CONTEXT, acknowledge this limitation politely.
Always cite your sources by referencing the document numbers when providing information from the context.
"""
        
        # Process chat history
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            # Add previous messages for context, up to a reasonable limit
            for msg in chat_history[-5:]:  # Include last 5 messages at most
                messages.append(msg)
        
        # Add the context and current query
        messages.append({
            "role": "user", 
            "content": f"""Please answer this question based only on the provided context:
QUESTION: {query}

CONTEXT:
{context}"""
        })
        
        try:
            # Try to use the configured LLM from settings
            llm_settings = settings.llm_settings
            
            # Handle different LLM backends
            if llm_settings.get("api_key") and llm_settings.get("provider") == "openai":
                # OpenAI API
                import openai
                client = openai.OpenAI(api_key=llm_settings["api_key"])
                
                response = client.chat.completions.create(
                    model=llm_settings.get("model_name", "gpt-3.5-turbo"),
                    messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content, sources
                
            elif llm_settings.get("provider") == "anthropic" and llm_settings.get("anthropic_api_key"):
                # Anthropic Claude API
                import anthropic
                client = anthropic.Anthropic(api_key=llm_settings["anthropic_api_key"])
                
                response = client.messages.create(
                    model=llm_settings.get("model_name", "claude-3-sonnet-20240229"),
                    messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response.content[0].text, sources
                
            elif "host" in llm_settings:
                # Ollama or local API
                ollama_url = f"{llm_settings['host']}/api/chat"
                
                response = requests.post(
                    ollama_url,
                    json={
                        "model": llm_settings.get("model_name", "llama3"),
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False
                    },
                    timeout=90  # Longer timeout for generation
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "Error: No response from LLM"), sources
                else:
                    return f"Error connecting to LLM ({response.status_code}). Please check your configuration.", sources
                    
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return f"I encountered an error when generating a response: {str(e)}", sources
        
        # Fallback if no LLM is configured
        return "I found some relevant information in the knowledge base, but I'm not configured to generate a response. Please contact your administrator to set up LLM integration.", sources
    
    def process_conversational_query(
        self, 
        query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process a user query in a conversational manner.
        
        Args:
            query: User query
            chat_history: Optional chat history for context
            temperature: Temperature for generation
            
        Returns:
            Dict with response, sources, and retrieved context
        """
        # Get retrieval context
        context = self.get_retrieval_context(query)
        
        # Generate response with LLM
        response, sources = self.generate_response(query, chat_history, temperature)
        
        return {
            "response": response,
            "sources": sources,
            "context": context
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary of collection statistics
        """
        try:
            stats = self.vector_db.get_collection_info(self.collection_name)
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}", exc_info=True)
            return {"error": str(e)}