#!/usr/bin/env python3
"""
Script to run the document ingestion process.
"""
import argparse
import sys
import os
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragstack.data_ingestion.file_watcher import FileWatcher
from ragstack.utils.logging import setup_logger
from ragstack.config.settings import settings


# Set logging level from environment or command line
debug_mode = os.environ.get("DEBUG_LOGGING", "0") == "1"
logger = setup_logger("run_ingestion", level="DEBUG" if debug_mode else None)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run document ingestion process")
    parser.add_argument(
        "--watch", action="store_true", help="Watch for new files continuously"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--file", type=str, help="Process a specific file only (path must be absolute)"
    )
    return parser.parse_args()


def print_system_info():
    """Print system information for debugging."""
    logger.info("--------- System Information ---------")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Data directories:")
    logger.info(f"  - Raw data: {settings.raw_data_dir}")
    logger.info(f"  - Processed data: {settings.processed_data_dir}")
    logger.info(f"  - Failed data: {settings.failed_data_dir}")
    logger.info(f"Log directory: {settings.logs_dir}")
    logger.info(f"Vector DB: {settings.vector_db_type}")
    logger.info(f"Embedding model: {settings.embedding_model['name']}")
    
    # Check for available files
    try:
        raw_files = list(os.listdir(settings.raw_data_dir))
        logger.info(f"Files in raw directory ({len(raw_files)}): {', '.join(raw_files[:5])}")
        if len(raw_files) > 5:
            logger.info(f"  ... and {len(raw_files) - 5} more")
    except Exception as e:
        logger.error(f"Error listing raw files: {e}")
    
    logger.info("---------------------------------------")


def check_external_services():
    """Check connectivity to external services."""
    logger.info("Checking external service connectivity...")
    
    # Check vector DB connectivity
    from ragstack.vector_db.factory import get_vector_db
    vector_db_type = settings.vector_db_type
    try:
        logger.info(f"Testing connection to {vector_db_type} vector database...")
        start_time = time.time()
        vector_db = get_vector_db(vector_db_type)
        elapsed = time.time() - start_time
        logger.info(f"Vector DB connection successful! ({elapsed:.2f}s)")
    except Exception as e:
        logger.error(f"Failed to connect to {vector_db_type}: {e}")
    
    # Check embedding model connectivity
    try:
        from ragstack.models.embeddings import get_embeddings_model
        logger.info("Testing connection to embedding model...")
        start_time = time.time()
        embedding_model = get_embeddings_model()
        
        # Try a simple embedding to check if it works
        try:
            test_embedding = embedding_model.encode_query("Test query for embedding model check")
            embedding_size = len(test_embedding)
            elapsed = time.time() - start_time
            logger.info(f"Embedding model check successful! Size: {embedding_size} ({elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"Failed to generate test embedding: {e}")
            
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
    
    logger.info("External service check completed")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set debug mode if specified as command line argument
    if args.debug:
        os.environ["DEBUG_LOGGING"] = "1"
        logger.setLevel("DEBUG")
        logger.info("Debug logging enabled via command line argument")
    
    try:
        # Print system information
        print_system_info()
        
        # Check connectivity to external services
        check_external_services()
        
        # Process specific file if requested
        if args.file:
            from ragstack.data_ingestion.document_processor import DocumentProcessor
            
            logger.info(f"Processing single file: {args.file}")
            
            if not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}")
                return 1
            
            processor = DocumentProcessor()
            success = processor.process_file(args.file)
            
            if success:
                logger.info(f"Successfully processed file: {args.file}")
            else:
                logger.error(f"Failed to process file: {args.file}")
                return 1
            
            return 0
        
        # Create file watcher
        file_watcher = FileWatcher()
        
        if args.watch:
            logger.info("Starting document ingestion service in watch mode")
            file_watcher.run_forever()
        else:
            logger.info("Processing existing documents")
            start_time = time.time()
            file_watcher.process_existing_files()
            elapsed = time.time() - start_time
            logger.info(f"Finished processing existing documents in {elapsed:.2f} seconds")
    
    except KeyboardInterrupt:
        logger.info("Stopping document ingestion service due to keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in document ingestion service: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())