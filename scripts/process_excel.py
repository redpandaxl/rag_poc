#!/usr/bin/env python
"""
Script to process Excel files using the specialized Excel processor.
"""
import os
import sys
import argparse
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragstack.data_ingestion.excel_processor import ExcelProcessor
from ragstack.vector_db.factory import get_vector_db
from ragstack.models.embeddings import get_embeddings_model
from ragstack.utils.logging import setup_logger
from ragstack.config.settings import settings

# Initialize logger
logger = setup_logger("process_excel", level="DEBUG")

def process_excel_file(
    file_path: str,
    strategy: str = "table",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    max_rows_per_chunk: int = 100,
    collection_name: str = None,
    vector_db_type: str = None,
    save_to_file: bool = True,
    skip_vector_db: bool = False,
) -> bool:
    """
    Process an Excel file and store it in the vector database.
    
    Args:
        file_path: Path to the Excel file
        strategy: Processing strategy ("table", "sheet", or "generic")
        chunk_size: Maximum size for text chunks
        chunk_overlap: Overlap between chunks
        max_rows_per_chunk: Maximum rows per chunk
        collection_name: Vector DB collection name (or use default)
        vector_db_type: Vector DB type (or use default)
        save_to_file: Whether to save chunks to text files
        skip_vector_db: Skip vector database insertion (useful for testing)
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        start_time = time.time()
        logger.info(f"Processing Excel file: {file_path}")
        
        # Create the Excel processor
        processor = ExcelProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_rows_per_chunk=max_rows_per_chunk,
            strategy=strategy,
        )
        
        # Process the file to get chunks
        chunks, chunk_metadata = processor.process_excel_file(Path(file_path))
        logger.info(f"Generated {len(chunks)} chunks from Excel file")
        
        if not chunks:
            logger.warning("No chunks were generated, nothing to save")
            return False
        
        # Save chunks to text files if requested
        if save_to_file:
            chunks_dir = Path(settings.processed_data_dir) / "chunks" / Path(file_path).stem
            os.makedirs(chunks_dir, exist_ok=True)
            
            # Save individual chunks to files
            for i, (chunk, metadata) in enumerate(zip(chunks, chunk_metadata)):
                chunk_file = chunks_dir / f"chunk_{i+1:03d}.txt"
                metadata_file = chunks_dir / f"chunk_{i+1:03d}_metadata.json"
                
                # Save chunk text
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                    
                # Save metadata
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                    
            logger.info(f"Saved {len(chunks)} chunks to {chunks_dir}")
            
        # Skip vector database if requested
        if skip_vector_db:
            logger.info("Skipping vector database insertion as requested")
        else:
            try:
                # Get the vector database
                vector_db_type = vector_db_type or settings.vector_db_type
                vector_db = get_vector_db(vector_db_type)
                logger.info(f"Using vector database: {vector_db_type}")
                
                # Get embedding model
                embedding_model = get_embeddings_model()
                logger.info(f"Using embedding model: {embedding_model.__class__.__name__}")
                
                # Generate embeddings
                logger.info("Generating embeddings for chunks")
                embeddings = []
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                    embedding = embedding_model.encode_query(chunk)
                    embeddings.append(embedding)
                
                # Ensure collection exists
                collection_name = collection_name or settings.vector_db_settings[vector_db_type]["collection_name"]
                vector_size = settings.embedding_model["vector_size"]
                
                if not vector_db.collection_exists(collection_name):
                    logger.info(f"Creating collection: {collection_name}")
                    vector_db.create_collection(collection_name, vector_size)
                
                # Insert into vector database
                logger.info(f"Inserting {len(chunks)} chunks into vector database collection: {collection_name}")
                vector_db.insert(
                    collection_name=collection_name,
                    vectors=embeddings,
                    documents=chunks,
                    metadata=chunk_metadata
                )
            except Exception as e:
                logger.error(f"Vector database or embedding error: {e}")
                logger.warning("Continuing with file processing despite vector DB error")
        
        end_time = time.time()
        logger.info(f"Successfully processed Excel file in {end_time - start_time:.2f} seconds")
        
        # Move file to processed directory if successful
        if os.path.exists(file_path):
            processed_path = os.path.join(
                settings.processed_data_dir, 
                os.path.basename(file_path)
            )
            logger.info(f"Moving file to processed directory: {processed_path}")
            os.makedirs(settings.processed_data_dir, exist_ok=True)
            os.rename(file_path, processed_path)
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}", exc_info=True)
        
        # Move to failed directory if the file exists
        if os.path.exists(file_path):
            failed_path = os.path.join(
                settings.failed_data_dir, 
                os.path.basename(file_path)
            )
            logger.info(f"Moving file to failed directory: {failed_path}")
            os.makedirs(settings.failed_data_dir, exist_ok=True)
            os.rename(file_path, failed_path)
            
        return False

def main():
    """Main function to process Excel files."""
    parser = argparse.ArgumentParser(description="Process Excel files with specialized processor")
    parser.add_argument("file_path", help="Path to the Excel file to process")
    parser.add_argument(
        "--strategy", 
        choices=["table", "sheet", "generic"], 
        default="table",
        help="Processing strategy"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=512,
        help="Maximum size for text chunks"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=50,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--max-rows", 
        type=int, 
        default=100,
        help="Maximum rows per chunk"
    )
    parser.add_argument(
        "--collection", 
        help="Vector DB collection name (or use default)"
    )
    parser.add_argument(
        "--vector-db", 
        help="Vector DB type (or use default)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save chunks to files"
    )
    parser.add_argument(
        "--skip-vector-db", 
        action="store_true",
        help="Skip vector database insertion"
    )
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        os.environ["DEBUG_LOGGING"] = "1"
    
    # Process the file
    success = process_excel_file(
        args.file_path,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_rows_per_chunk=args.max_rows,
        collection_name=args.collection,
        vector_db_type=args.vector_db,
        save_to_file=not args.no_save,
        skip_vector_db=args.skip_vector_db,
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()