#!/usr/bin/env python
"""
Test script for debugging the text chunking process.
"""
import os
import sys
import logging
import time
import argparse
import traceback
from pathlib import Path
import tracemalloc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragstack.data_ingestion.text_chunker import TextChunker, ChunkingStrategy
from ragstack.utils.logging import setup_logger

# Initialize logger with debug level
os.environ["DEBUG_LOGGING"] = "1"
logger = setup_logger("test_chunking", level="DEBUG")

def test_file_chunking(file_path: str, strategy: str = "recursive", 
                       chunk_size: int = 512, chunk_overlap: int = 50,
                       max_recursion_depth: int = 5):
    """
    Test chunking on a specific file with detailed logging and memory tracking.
    
    Args:
        file_path: Path to the file to test
        strategy: Chunking strategy (recursive, fixed_size, sentence, paragraph)
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        max_recursion_depth: Maximum recursion depth for recursive chunking
    """
    # Start memory tracking
    tracemalloc.start()
    
    chunker = TextChunker()
    
    # Validate file path
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Testing chunking on file: {file_path}")
    logger.info(f"Strategy: {strategy}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
    
    # Map strategy string to enum
    strategy_map = {
        "recursive": ChunkingStrategy.RECURSIVE,
        "fixed_size": ChunkingStrategy.FIXED_SIZE,
        "sentence": ChunkingStrategy.SENTENCE,
        "paragraph": ChunkingStrategy.PARAGRAPH
    }
    
    if strategy not in strategy_map:
        logger.error(f"Invalid strategy: {strategy}. Must be one of {', '.join(strategy_map.keys())}")
        return
    
    chunking_strategy = strategy_map[strategy]
    
    # Read the file
    try:
        logger.info(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        logger.info(f"File content length: {len(content)} characters")
        
        # Memory snapshot before chunking
        memory_before = tracemalloc.get_traced_memory()
        logger.info(f"Memory before chunking: {memory_before[0] / (1024*1024):.2f} MB, peak: {memory_before[1] / (1024*1024):.2f} MB")
        
        # Time the chunking process
        logger.info("Starting chunking process...")
        start_time = time.time()
        
        # Set recursion limit for debugging
        if strategy == "recursive":
            import sys
            original_limit = sys.getrecursionlimit()
            logger.info(f"Current recursion limit: {original_limit}")
            # Increase limit if needed for testing
            new_limit = max(1000, original_limit)
            sys.setrecursionlimit(new_limit)
            logger.info(f"Set recursion limit to: {new_limit}")
        
        # Execute chunking with try-except for detailed error reporting
        try:
            chunks = chunker.chunk_text(
                content,
                strategy=chunking_strategy, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                max_recursion_depth=max_recursion_depth
            )
            
            # Calculate and log timing
            chunking_time = time.time() - start_time
            logger.info(f"Chunking completed in {chunking_time:.2f} seconds")
            
            # Get memory usage after chunking
            memory_after = tracemalloc.get_traced_memory()
            logger.info(f"Memory after chunking: {memory_after[0] / (1024*1024):.2f} MB, peak: {memory_after[1] / (1024*1024):.2f} MB")
            
            # Log chunk statistics
            if chunks:
                chunk_sizes = [len(chunk) for chunk in chunks]
                avg_chunk_size = sum(chunk_sizes) / len(chunks) if chunks else 0
                min_chunk_size = min(chunk_sizes) if chunks else 0
                max_chunk_size = max(chunk_sizes) if chunks else 0
                
                logger.info(f"Generated {len(chunks)} chunks")
                logger.info(f"Chunk statistics - Average size: {avg_chunk_size:.1f}, Min: {min_chunk_size}, Max: {max_chunk_size}")
                
                # Log first few chunks for debugging
                for i, chunk in enumerate(chunks[:3]):
                    logger.debug(f"Chunk {i+1} (length: {len(chunk)}): {chunk[:100]}...")
                
                if len(chunks) > 3:
                    logger.debug(f"... and {len(chunks) - 3} more chunks")
            else:
                logger.warning("No chunks were generated!")
                
        except RecursionError as e:
            logger.error(f"Recursion error during chunking: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
        except Exception as e:
            logger.error(f"Error during chunking: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
        finally:
            # Restore recursion limit if changed
            if strategy == "recursive":
                sys.setrecursionlimit(original_limit)
                logger.info(f"Restored recursion limit to: {original_limit}")
                
    except Exception as e:
        logger.error(f"Error reading or processing file: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
    
    # Stop memory tracking
    tracemalloc.stop()

def main():
    """Parse arguments and run the chunking test."""
    parser = argparse.ArgumentParser(description="Test and debug the text chunking process")
    parser.add_argument("file_path", help="Path to the file to test chunking on")
    parser.add_argument("--strategy", "-s", default="recursive", 
                        choices=["recursive", "fixed_size", "sentence", "paragraph"],
                        help="Chunking strategy to use")
    parser.add_argument("--chunk-size", "-c", type=int, default=512,
                        help="Maximum size of each chunk")
    parser.add_argument("--chunk-overlap", "-o", type=int, default=50,
                        help="Number of characters to overlap between chunks")
    parser.add_argument("--max-recursion-depth", "-r", type=int, default=5,
                        help="Maximum recursion depth for recursive chunking")
    
    args = parser.parse_args()
    
    test_file_chunking(
        args.file_path,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_recursion_depth=args.max_recursion_depth
    )

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())