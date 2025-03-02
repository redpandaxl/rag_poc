#!/usr/bin/env python3
"""
Script to run the web interface and API.
"""
import argparse
import logging
import multiprocessing
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragstack.web.api import start_api
from ragstack.web.streamlit_app import start_streamlit
from ragstack.utils.logging import setup_logger


logger = setup_logger("run_web")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run web interface and API")
    parser.add_argument(
        "--api-only", action="store_true", help="Run only the API, not the Streamlit UI"
    )
    parser.add_argument(
        "--ui-only", action="store_true", help="Run only the Streamlit UI, not the API"
    )
    return parser.parse_args()


def run_api():
    """Run the FastAPI server."""
    try:
        logger.info("Starting API server")
        start_api()
    except Exception as e:
        logger.error(f"Error in API server: {e}", exc_info=True)


def run_streamlit():
    """Run the Streamlit app."""
    try:
        logger.info("Starting Streamlit app")
        start_streamlit()
    except Exception as e:
        logger.error(f"Error in Streamlit app: {e}", exc_info=True)


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.api_only and args.ui_only:
            logger.error("Cannot specify both --api-only and --ui-only")
            return 1
        
        if args.api_only:
            # Run API only
            run_api()
        elif args.ui_only:
            # Run UI only
            run_streamlit()
            # Keep main process alive
            while True:
                time.sleep(10)
        else:
            # Run both API and UI in separate processes
            api_process = multiprocessing.Process(target=run_api)
            api_process.start()
            
            # Sleep to allow API to start
            time.sleep(2)
            
            ui_process = multiprocessing.Process(target=run_streamlit)
            ui_process.start()
            
            # Wait for processes to complete
            api_process.join()
            ui_process.join()
    
    except KeyboardInterrupt:
        logger.info("Stopping web services due to keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in web services: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())