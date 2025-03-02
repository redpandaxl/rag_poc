"""
File watching service for the RAG application.
"""
import os
import time
from pathlib import Path
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ragstack.config.settings import settings
from ragstack.data_ingestion.document_processor import DocumentProcessor
from ragstack.utils.logging import setup_logger


logger = setup_logger("ragstack.data_ingestion.file_watcher")


class DocumentHandler(FileSystemEventHandler):
    """Watchdog handler for processing new documents."""
    
    def __init__(self, processor: DocumentProcessor):
        """
        Initialize document handler.
        
        Args:
            processor: Document processor instance
        """
        self.processor = processor
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            logger.info(f"New file detected: {event.src_path}")
            # Add a small delay to ensure the file is fully written
            import time
            time.sleep(0.5)
            # Check if the file still exists before processing
            if os.path.exists(event.src_path):
                self.processor.process_file(event.src_path)
            else:
                logger.warning(f"File {event.src_path} disappeared before processing could start")
    
    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory and event.dest_path.startswith(str(settings.raw_data_dir)):
            logger.info(f"File moved into watched directory: {event.dest_path}")
            # Add a small delay to ensure the file is fully moved
            import time
            time.sleep(0.5)
            # Check if the file still exists before processing
            if os.path.exists(event.dest_path):
                self.processor.process_file(event.dest_path)
            else:
                logger.warning(f"File {event.dest_path} disappeared before processing could start")


class FileWatcher:
    """Watch directory for new files and process them."""
    
    def __init__(
        self, 
        watch_dir: Optional[Path] = None,
        processor: Optional[DocumentProcessor] = None
    ):
        """
        Initialize file watcher.
        
        Args:
            watch_dir: Directory to watch for new files
            processor: Document processor instance
        """
        self.watch_dir = watch_dir or settings.raw_data_dir
        self.processor = processor or DocumentProcessor()
        self.observer = None
    
    def start(self):
        """Start watching for new files."""
        # Process existing files
        self.process_existing_files()
        
        # Start file watcher
        self.observer = Observer()
        handler = DocumentHandler(self.processor)
        self.observer.schedule(handler, self.watch_dir, recursive=False)
        self.observer.start()
        
        logger.info(f"Started file watcher for directory: {self.watch_dir}")
    
    def process_existing_files(self):
        """Process any existing files in the watch directory."""
        processed_count = 0
        skipped_count = 0
        
        # Get a list of all files before processing
        files_to_process = []
        for file_path in os.listdir(self.watch_dir):
            full_path = os.path.join(self.watch_dir, file_path)
            if os.path.isfile(full_path):
                files_to_process.append(full_path)
        
        logger.info(f"Found {len(files_to_process)} existing files to process")
        
        # Process each file
        for full_path in files_to_process:
            if os.path.exists(full_path) and os.path.isfile(full_path):
                logger.info(f"Processing existing file: {full_path}")
                result = self.processor.process_file(full_path)
                if result:
                    processed_count += 1
                else:
                    skipped_count += 1
            else:
                logger.warning(f"File {full_path} no longer exists, skipping")
                skipped_count += 1
        
        logger.info(f"Processed {processed_count} existing files, skipped {skipped_count} files")
    
    def stop(self):
        """Stop watching for new files."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped file watcher")
    
    def run_forever(self):
        """Run the file watcher indefinitely."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping file watcher due to keyboard interrupt")
        finally:
            self.stop()