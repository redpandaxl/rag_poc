"""
Text chunking utilities for the RAG application.
"""
from typing import List, Optional, Callable
from enum import Enum, auto

import nltk
from nltk.tokenize import sent_tokenize

from ragstack.config.settings import settings
from ragstack.utils.logging import setup_logger


class ChunkingStrategy(str, Enum):
    """Enumeration of available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"


logger = setup_logger("ragstack.data_ingestion.text_chunker")


# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt resource")
    nltk.download('punkt')


class TextChunker:
    """Text chunking strategies for document processing."""
    
    @staticmethod
    def simple_chunk(
        text: str, 
        chunk_size: Optional[int] = None, 
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        Chunk text into smaller pieces with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum character length for each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Use settings if not provided
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        
        logger.debug(f"Created {len(chunks)} simple chunks from {len(text)} characters")
        return chunks
    
    @staticmethod
    def semantic_chunk(
        text: str, 
        chunk_size: Optional[int] = None
    ) -> List[str]:
        """
        Chunk text into smaller pieces based on sentences.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum character length for each chunk
            
        Returns:
            List of text chunks preserving sentence boundaries
        """
        # Use settings if not provided
        chunk_size = chunk_size or settings.chunk_size
        
        sentences = sent_tokenize(text)
        chunks = []
        chunk = ""
        
        for sentence in sentences:
            if len(chunk) + len(sentence) < chunk_size:
                chunk += sentence + " "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + " "
        
        # Add the last chunk if not empty
        if chunk:
            chunks.append(chunk.strip())
        
        logger.debug(f"Created {len(chunks)} semantic chunks from {len(text)} characters")
        return chunks
    
    @staticmethod
    def recursive_chunk(
        text: str, 
        chunk_size: Optional[int] = None, 
        overlap: Optional[int] = None,
        max_depth: int = 2,  # Reduced max recursion depth to prevent stack overflow
        _current_depth: int = 0  # Internal parameter to track current recursion depth
    ) -> List[str]:
        """
        Chunk text into smaller pieces recursively.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum character length for each chunk
            overlap: Number of characters to overlap between chunks
            max_depth: Maximum recursion depth to prevent stack overflow
            
        Returns:
            List of text chunks
        """
        # Emergency safety checks first
        if text is None:
            logger.error("Received None text in recursive_chunk")
            return ["EMPTY CONTENT"]
            
        if not isinstance(text, str):
            logger.error(f"Received non-string in recursive_chunk: {type(text)}")
            try:
                text = str(text)
            except:
                return ["INVALID CONTENT TYPE"]
        
        # Use settings if not provided
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap
        
        # Ensure valid parameters to prevent errors
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            logger.error(f"Invalid chunk_size: {chunk_size}, using default 512")
            chunk_size = 512
            
        if not isinstance(overlap, int) or overlap < 0:
            logger.error(f"Invalid overlap: {overlap}, using default 50")
            overlap = 50
        
        # Prevent excessive recursion by limiting the maximum recursion depth
        if max_depth > 3:
            max_depth = 3
            logger.warning(f"Limiting max_depth to 3 to prevent stack overflow")
        
        start_time = __import__('time').time()
        
        logger.debug(f"Recursive chunking called - Text length: {len(text)}, depth: {max_depth}")
        
        # Base case: text is small enough
        if len(text) < chunk_size:
            logger.debug(f"Text already small enough ({len(text)} < {chunk_size}), returning as is")
            return [text]
        
        # Base case: max recursion depth reached, fall back to simple chunking
        if _current_depth >= max_depth:
            logger.warning(f"Maximum recursion depth {max_depth} reached at depth {_current_depth}, falling back to simple chunking for text of length {len(text)}")
            try:
                simple_chunks = TextChunker.simple_chunk(text, chunk_size, overlap)
                logger.debug(f"Simple chunking produced {len(simple_chunks)} chunks as fallback")
                return simple_chunks
            except Exception as e:
                logger.error(f"Simple chunking failed: {e}, falling back to emergency chunking")
                # Emergency chunking - just split by chunk size
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
        
        # Try semantic chunking first - with NLTK pre-check
        try:
            logger.debug("Attempting sentence tokenization")
            
            # Verify NLTK resources are available
            try:
                nltk_resource_check = nltk.data.find('tokenizers/punkt')
                logger.debug(f"NLTK punkt resource found at: {nltk_resource_check}")
            except LookupError:
                logger.warning("NLTK punkt resource not found, downloading...")
                try:
                    nltk.download('punkt')
                    logger.info("NLTK punkt resource downloaded successfully")
                except Exception as nltk_error:
                    logger.error(f"Failed to download NLTK punkt: {nltk_error}, falling back to simple chunking")
                    simple_chunks = TextChunker.simple_chunk(text, chunk_size, overlap)
                    return simple_chunks
                
            sentences = sent_tokenize(text)
            logger.debug(f"Sentence tokenization successful - found {len(sentences)} sentences")
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}, falling back to simple chunking")
            try:
                simple_chunks = TextChunker.simple_chunk(text, chunk_size, overlap)
                logger.debug(f"Simple chunking produced {len(simple_chunks)} chunks as fallback")
                return simple_chunks
            except Exception as e2:
                logger.error(f"Simple chunking also failed: {e2}, falling back to emergency chunking")
                # Emergency chunking - just split by chunk size
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
        
        # If no sentences found or very long text without sentence breaks,
        # fall back to simple chunking
        if not sentences or len(sentences) == 1:
            logger.debug(f"No sentence breaks found, falling back to simple chunking for text of length {len(text)}")
            try:
                simple_chunks = TextChunker.simple_chunk(text, chunk_size, overlap)
                logger.debug(f"Simple chunking produced {len(simple_chunks)} chunks as fallback")
                return simple_chunks
            except Exception as e:
                logger.error(f"Simple chunking failed: {e}, falling back to emergency chunking")
                # Emergency chunking - just split by chunk size
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
        
        logger.debug(f"Building chunks from {len(sentences)} sentences")    
        chunks = []
        chunk = ""
        
        for sentence in sentences:
            if len(chunk) + len(sentence) < chunk_size:
                chunk += sentence + " "
            else:
                if chunk.strip():
                    chunks.append(chunk.strip())
                chunk = sentence + " "
        
        # Add the last chunk if not empty
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # If semantic chunking didn't produce any chunks or produced very large chunks
        # fall back to simple chunking
        if not chunks or (len(chunks) == 1 and len(chunks[0]) > chunk_size):
            logger.debug(f"Semantic chunking failed to produce appropriate chunks, falling back to simple chunking")
            try:
                simple_chunks = TextChunker.simple_chunk(text, chunk_size, overlap)
                logger.debug(f"Simple chunking produced {len(simple_chunks)} chunks as fallback")
                return simple_chunks
            except Exception as e:
                logger.error(f"Simple chunking failed: {e}, falling back to emergency chunking")
                # Emergency chunking - just split by chunk size
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
        
        # Recursively process each chunk with decremented max_depth
        logger.debug(f"Semantic chunking produced {len(chunks)} initial chunks, processing recursively")
        result = []
        large_chunk_count = 0
        
        # Iterative approach instead of recursive for large chunk count to prevent stack overflow
        if len(chunks) > 20:
            logger.warning(f"Large number of chunks ({len(chunks)}), using non-recursive approach")
            for i, chunk in enumerate(chunks):
                if len(chunk) > chunk_size:
                    # Use simple chunking instead of recursion for large chunk counts
                    simple_sub_chunks = TextChunker.simple_chunk(chunk, chunk_size, overlap)
                    result.extend(simple_sub_chunks)
                    large_chunk_count += 1
                else:
                    result.append(chunk)
        else:
            # Standard approach for reasonable chunk counts
            for i, chunk in enumerate(chunks):
                # Only recurse if the chunk is still too large
                if len(chunk) > chunk_size:
                    large_chunk_count += 1
                    logger.debug(f"Recursing on chunk {i+1}/{len(chunks)} with length {len(chunk)} > {chunk_size}, depth: {_current_depth+1}/{max_depth}")
                    try:
                        sub_chunks = TextChunker.recursive_chunk(
                            chunk, 
                            chunk_size, 
                            overlap, 
                            max_depth=max_depth,
                            _current_depth=_current_depth + 1
                        )
                        logger.debug(f"Recursion produced {len(sub_chunks)} sub-chunks for chunk {i+1}")
                        result.extend(sub_chunks)
                    except RecursionError as e:
                        logger.error(f"Recursion error processing chunk {i+1}: {e}, falling back to simple chunking")
                        try:
                            simple_chunks = TextChunker.simple_chunk(chunk, chunk_size, overlap)
                            logger.debug(f"Simple chunking produced {len(simple_chunks)} chunks as recursion error fallback")
                            result.extend(simple_chunks)
                        except Exception as e2:
                            logger.error(f"Simple chunking also failed: {e2}, using emergency chunking")
                            # Emergency chunking - just split by chunk size
                            result.extend([chunk[i:i+chunk_size] for i in range(0, len(chunk), chunk_size-overlap)])
                    except Exception as e:
                        logger.error(f"Error in recursive chunking for chunk {i+1}: {e}, falling back to simple chunking")
                        try:
                            simple_chunks = TextChunker.simple_chunk(chunk, chunk_size, overlap)
                            result.extend(simple_chunks)
                        except:
                            # Last resort emergency chunking
                            result.extend([chunk[i:i+chunk_size] for i in range(0, len(chunk), chunk_size-overlap)])
                else:
                    logger.debug(f"Chunk {i+1}/{len(chunks)} with length {len(chunk)} <= {chunk_size} added as is")
                    result.append(chunk)
        
        end_time = __import__('time').time()
        logger.debug(f"Created {len(result)} recursive chunks in {end_time - start_time:.2f}s from {len(text)} chars, {large_chunk_count} needed further recursion")
        
        # Final safety check
        if not result:
            logger.error("No chunks created, falling back to emergency chunking")
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
            
        return result
    
    def chunk_text(
        self,
        text: str, 
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None,
        max_recursion_depth: int = 3,
        minimum_chunk_size: Optional[int] = None
    ) -> List[str]:
        """
        Chunk text using the specified strategy.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy to use
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            max_recursion_depth: Maximum recursion depth for recursive chunking
            minimum_chunk_size: Minimum chunk size
            
        Returns:
            List of text chunks
        """
        # Use settings if not provided
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        if not text:
            logger.warning("Empty text provided to chunk_text")
            return []
            
        logger.debug(f"Chunking text of length {len(text)} with strategy {strategy}")
        
        # Map strategy to method
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self.simple_chunk(text, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self.semantic_chunk(text, chunk_size)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            # Split by paragraphs (double newlines)
            paragraphs = text.split("\n\n")
            # Filter out empty paragraphs
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            if not paragraphs:
                logger.warning("No paragraphs found, falling back to simple chunking")
                return self.simple_chunk(text, chunk_size, chunk_overlap)
                
            # If paragraphs are too large, chunk them further
            result = []
            for p in paragraphs:
                if len(p) > chunk_size:
                    # Use semantic chunking for paragraphs
                    p_chunks = self.semantic_chunk(p, chunk_size)
                    result.extend(p_chunks)
                else:
                    result.append(p)
            return result
        elif strategy == ChunkingStrategy.RECURSIVE:
            return self.recursive_chunk(text, chunk_size, chunk_overlap, max_recursion_depth)
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}, falling back to recursive")
            return self.recursive_chunk(text, chunk_size, chunk_overlap, max_recursion_depth)
        
    @classmethod
    def get_chunking_method(cls, method_name: str) -> Callable[[str, Optional[int], Optional[int]], List[str]]:
        """
        Get chunking method by name.
        
        Args:
            method_name: Name of the chunking method
            
        Returns:
            Chunking method function
            
        Raises:
            ValueError: If the chunking method is unsupported
        """
        methods = {
            "simple": cls.simple_chunk,
            "semantic": cls.semantic_chunk,
            "recursive": cls.recursive_chunk
        }
        
        if method_name not in methods:
            logger.warning(f"Unknown chunking method: {method_name}, using recursive")
            return cls.recursive_chunk
        
        return methods[method_name]