#!/usr/bin/env python3
"""
Script to retry processing failed files.
"""
import argparse
import sys
import os
import shutil
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragstack.data_ingestion.document_processor import DocumentProcessor
from ragstack.utils.logging import setup_logger
from ragstack.config.settings import settings

# Set up logger
logger = setup_logger("retry_failed_files", level="DEBUG")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Retry processing failed files")
    parser.add_argument("--all", action="store_true", help="Retry all failed files")
    parser.add_argument("--file", type=str, help="Retry a specific failed file")
    parser.add_argument("--extension", type=str, help="Retry all files with a specific extension (e.g., '.pdf')")
    args = parser.parse_args()
    
    failed_dir = settings.failed_data_dir
    raw_dir = settings.raw_data_dir
    
    # Get list of failed files
    failed_files = [f for f in os.listdir(failed_dir) if os.path.isfile(os.path.join(failed_dir, f))]
    
    if not failed_files:
        logger.info("No failed files found.")
        return 0
    
    logger.info(f"Found {len(failed_files)} failed files.")
    
    # Filter files based on arguments
    if args.file:
        target_files = [args.file] if args.file in failed_files else []
        if not target_files:
            logger.error(f"File not found in failed directory: {args.file}")
            return 1
    elif args.extension:
        target_files = [f for f in failed_files if f.endswith(args.extension)]
        logger.info(f"Found {len(target_files)} failed files with extension {args.extension}")
    elif args.all:
        target_files = failed_files
    else:
        logger.error("Please specify --all, --file, or --extension")
        return 1
    
    # Create document processor
    processor = DocumentProcessor()
    
    # Process each file
    success_count = 0
    failed_count = 0
    
    for filename in target_files:
        failed_path = os.path.join(failed_dir, filename)
        raw_path = os.path.join(raw_dir, filename)
        
        logger.info(f"Moving {failed_path} to {raw_path} for reprocessing")
        try:
            # Copy file back to raw directory
            shutil.copy2(failed_path, raw_path)
            
            # Process the file
            logger.info(f"Processing {raw_path}")
            result = processor.process_file(raw_path)
            
            if result:
                logger.info(f"Successfully processed {filename}")
                success_count += 1
                
                # Remove the file from failed directory if it was successfully processed
                os.remove(failed_path)
            else:
                logger.error(f"Failed to process {filename}")
                failed_count += 1
        except Exception as e:
            logger.error(f"Error while processing {filename}: {e}")
            failed_count += 1
    
    logger.info(f"Retry summary: {success_count} succeeded, {failed_count} failed")
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())