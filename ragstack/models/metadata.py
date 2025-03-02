"""
Metadata generation utilities for the RAG application.
"""
import json
import re
import time
import uuid
import random
from typing import Dict, Any, Optional

from ollama import Client

from ragstack.config.settings import settings
from ragstack.utils.logging import setup_logger

# Mock client for when Ollama is not available
class MockOllamaClient:
    """Mock Ollama client that generates simple metadata."""
    
    def list(self) -> Dict[str, Any]:
        """Mock list models response."""
        return {"models": [{"name": "mock-model"}]}
    
    def generate(self, model: str, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Generate metadata using a simple template."""
        sample_titles = ["Project Overview", "Technical Documentation", "Meeting Notes", 
                        "Product Roadmap", "Market Analysis"]
        sample_tags = ["business", "technology", "planning", "research", "development", 
                      "marketing", "operations", "finance", "strategy", "product"]
        
        # Try to extract document content from the prompt
        document_text = ""
        if "Document:" in prompt:
            document_parts = prompt.split("Document:", 1)
            if len(document_parts) > 1:
                document_text = document_parts[1]
        
        words = document_text.split() if document_text else prompt.split()
        
        # Get up to 15 words for the summary, but handle short texts
        if len(words) > 15:
            summary_words = random.sample(words, 15)
        else:
            summary_words = words
        
        summary = " ".join(summary_words) if summary_words else "content not available"
        
        # Create a mock response
        metadata = {
            "title": random.choice(sample_titles),
            "tags": random.sample(sample_tags, 3),
            "summary": f"This document appears to be about {summary}..."
        }
        
        logger.info(f"Mock metadata generator created: {metadata}")
        return {"response": json.dumps(metadata)}


logger = setup_logger("ragstack.models.metadata")


class MetadataGenerator:
    """Generate metadata for documents using LLM."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Initialize metadata generator.
        
        Args:
            host: Ollama API host
            model_name: Model name to use for generation
            max_retries: Maximum number of retries on failure
        """
        # Use settings if not provided
        llm_settings = settings.llm_settings
        self.host = host or llm_settings["host"]
        self.model_name = model_name or llm_settings["model_name"]
        self.max_retries = max_retries
        self.client = None
    
    def initialize(self) -> None:
        """Initialize the connection to Ollama."""
        try:
            logger.info(f"Connecting to Ollama at {self.host}")
            self.client = Client(host=self.host)
            # Test the connection
            logger.info(f"Testing Ollama connection...")
            # Try a simple test request
            try:
                # Simple test to see if Ollama is responding
                self.client.list()
                logger.info(f"Ollama client initialized and connected successfully")
            except Exception as test_error:
                logger.warning(f"Ollama connection test failed: {test_error}")
                logger.warning("Using mock metadata generator as fallback")
                self.client = MockOllamaClient()
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            logger.warning("Using mock metadata generator as fallback")
            self.client = MockOllamaClient()
            # Don't raise exception, use fallback instead
    
    def _extract_title(self, content: str) -> str:
        """
        Extract a reasonable title from content.
        
        Args:
            content: Document content
            
        Returns:
            Extracted title
        """
        # Try to extract from first line
        first_line = content.split('\n', 1)[0].strip()
        if 5 <= len(first_line) <= 100:
            return first_line
            
        # If first line isn't suitable, use first N words
        words = content.split()[:10]
        title = " ".join(words)
        if len(title) > 100:
            title = title[:97] + "..."
        return title
        
    def generate(self, content: str) -> Dict[str, Any]:
        """
        Use Ollama to generate metadata for the document.
        
        Args:
            content: Document content
            
        Returns:
            Metadata dictionary with title, tags, and summary
        """
        start_time = time.time()
        
        # Safety check for None or empty content
        if not content:
            logger.warning("Empty content provided to metadata generator")
            return {
                "title": "Empty Document",
                "tags": ["empty", "no-content"],
                "summary": "This document contains no text content."
            }
            
        # Initialize if not already initialized
        if self.client is None:
            try:
                self.initialize()
            except Exception as init_error:
                logger.error(f"Failed to initialize metadata generator: {init_error}")
                return self._generate_fallback_metadata(content)
        
        # Quick check for MockOllamaClient as our client
        if isinstance(self.client, MockOllamaClient):
            logger.debug("Using mock metadata generator")
            mock_response = self.client.generate(
                model=self.model_name, 
                prompt=f"Generate metadata for: {content[:100]}"
            )
            try:
                # Parse the JSON string in the response
                metadata_dict = json.loads(mock_response["response"])
                return metadata_dict
            except (json.JSONDecodeError, KeyError):
                # If parsing fails, return a basic fallback dictionary
                return self._generate_fallback_metadata(content)
        
        # Truncate content to avoid token limits (shorter than before)
        truncated_content = content[:1500]
        if len(content) > 1500:
            logger.debug(f"Content truncated from {len(content)} to 1500 chars for metadata generation")
        
        # Create prompt
        prompt = f"""
        You are an intelligent metadata generator. Analyze the provided document content and generate JSON metadata containing:
        - title: A concise title for the document
        - tags: A list of 3-5 relevant topics
        - summary: A brief summary (1-2 sentences)

        Return ONLY the JSON object without any extra text.

        Document: {truncated_content}

        Example JSON Output:
        {{
            "title": "Document Title",
            "tags": ["tag1", "tag2"],
            "summary": "A brief summary of the document."
        }}
        """
        
        # Try to generate metadata with retries
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Generating metadata (attempt {attempt+1}/{self.max_retries})")
                
                # Add timeout to avoid hanging
                timeout_seconds = 5
                logger.debug(f"Setting timeout to {timeout_seconds} seconds")
                
                try:
                    # Send request to Ollama with timeout handling
                    request_start = time.time()
                    response = self.client.generate(
                        model=self.model_name,
                        prompt=prompt,
                        stream=False  # Set stream to False to get the complete response
                    )
                    request_time = time.time() - request_start
                    logger.debug(f"Ollama request completed in {request_time:.2f}s")
                    
                except Exception as request_error:
                    logger.error(f"Ollama request failed: {request_error}")
                    
                    # Check if we should retry
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Retrying in 1 second (attempt {attempt+1}/{self.max_retries})...")
                        time.sleep(1)
                        continue
                    else:
                        # All retries exhausted
                        logger.error("Max retries exceeded, using fallback metadata")
                        return self._generate_fallback_metadata(content)
                
                # Extract JSON from response using regex
                json_pattern = r'\{.*?\}(?=\n|\Z)'
                match = re.search(json_pattern, response["response"], re.DOTALL)
                
                if match:
                    json_string = match.group(0)
                    try:
                        metadata = json.loads(json_string)
                        
                        # Validate and fix metadata
                        if not metadata:
                            logger.warning("Empty metadata returned")
                            metadata = {}
                            
                        # Ensure required fields exist
                        if "title" not in metadata or not metadata["title"]:
                            title = self._extract_title(content)
                            logger.debug(f"Missing title, using extracted title: {title}")
                            metadata["title"] = title
                            
                        if "tags" not in metadata or not isinstance(metadata["tags"], list):
                            logger.debug("Missing or invalid tags, using default tags")
                            metadata["tags"] = ["auto-generated"]
                            
                        if "summary" not in metadata or not metadata["summary"]:
                            logger.debug("Missing summary, using default")
                            metadata["summary"] = f"Document containing {len(content)} characters"
                        
                        total_time = time.time() - start_time
                        logger.debug(f"Successfully generated metadata in {total_time:.2f}s: {metadata}")
                        return metadata
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON: {e}")
                else:
                    logger.warning("No JSON found in LLM response")
                
                # If we get here, metadata generation failed but we got a response
                if attempt < self.max_retries - 1:
                    logger.warning(f"Retrying metadata generation in 1 second...")
                    time.sleep(1)
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Metadata generation error (attempt {attempt+1}/{self.max_retries}): {e}")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to generate metadata after {self.max_retries} attempts: {e}")
        
        # Return fallback metadata if all attempts failed
        return self._generate_fallback_metadata(content)
    
    def _generate_fallback_metadata(self, content: str) -> Dict[str, Any]:
        """
        Generate fallback metadata when LLM generation fails.
        
        Args:
            content: Document content
            
        Returns:
            Basic metadata dictionary
        """
        logger.warning("Using fallback metadata generation")
        
        # Extract a title from the content
        title = self._extract_title(content)
        
        # Get document stats for tags
        chars = len(content)
        words = len(content.split())
        lines = len(content.splitlines())
        
        # Create fallback metadata
        metadata = {
            "title": title,
            "tags": ["auto-generated", "fallback"],
            "summary": f"Document with {words} words across {lines} lines"
        }
        
        logger.debug(f"Fallback metadata generated: {metadata}")
        return metadata