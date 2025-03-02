#!/usr/bin/env python3
"""
RAGStack: Self-hosted RAG Environment

Main entry point for the application.
"""
import argparse
import logging
import sys

from ragstack.data_ingestion.file_watcher import FileWatcher
from ragstack.web.api import start_api
from ragstack.web.streamlit_app import start_streamlit
from ragstack.utils.logging import setup_logger


logger = setup_logger("main")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAGStack: Self-hosted RAG Environment")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingestion command
    ingest_parser = subparsers.add_parser("ingest", help="Run document ingestion")
    ingest_parser.add_argument(
        "--watch", action="store_true", help="Watch for new files continuously"
    )
    
    # Web command
    web_parser = subparsers.add_parser("web", help="Run web interface")
    web_parser.add_argument(
        "--api-only", action="store_true", help="Run only the API, not the Streamlit UI"
    )
    web_parser.add_argument(
        "--ui-only", action="store_true", help="Run only the Streamlit UI, not the API"
    )
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Add project root to Python path to find modules
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    if args.command == "ingest":
        # Run ingestion directly here to avoid import issues
        import multiprocessing
        from ragstack.data_ingestion.file_watcher import FileWatcher
        
        logger.info("Starting document ingestion service")
        file_watcher = FileWatcher()
        
        if args.watch:
            file_watcher.run_forever()
        else:
            file_watcher.process_existing_files()
        
    elif args.command == "web":
        # Run web interface directly here to avoid import issues
        import multiprocessing
        import time
        from ragstack.web.api import start_api
        from ragstack.web.streamlit_app import start_streamlit
        from ragstack.config.settings import settings
        
        processes = []
        
        # Handle API/UI only flags
        if args.api_only:
            logger.info("Starting API server only")
            start_api()
        elif args.ui_only:
            logger.info("Starting Streamlit UI only")
            start_streamlit()
        else:
            # Start API in a separate process
            logger.info("Starting API server")
            api_process = multiprocessing.Process(target=start_api)
            api_process.start()
            processes.append(api_process)
            
            # Give the API a moment to start
            time.sleep(2)
            
            # Print URLs
            logger.info(f"API running at: http://{settings.api_host}:{settings.api_port}")
            logger.info(f"Web UI will run at: http://{settings.web_host}:{settings.web_port}")
            
            # Run Streamlit in the main process
            logger.info("Starting Streamlit UI")
            start_streamlit()  # This will block until Streamlit exits
            
            # Clean up processes
            for process in processes:
                process.terminate()
        
    elif args.command == "version":
        # Show version information
        print(f"RAGStack version 0.1.0")
        
    else:
        # Show help if no command is provided
        logger.info("No command specified. Use --help for usage information.")
        print("RAGStack: Self-hosted RAG Environment")
        print("Usage: python main.py [ingest|web|version] [options]")
        print("       python main.py --help")


if __name__ == "__main__":
    main()
