#!/usr/bin/env python3
"""
Script to run the RAG chat interface.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path to enable imports
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from ragstack.web.chat_interface import start_chat_interface
from ragstack.utils.logging import setup_logger

# Set up logger
logger = setup_logger("run_chat")

if __name__ == "__main__":
    logger.info("Starting RAG Chat Interface")
    try:
        start_chat_interface()
    except KeyboardInterrupt:
        logger.info("Chat interface stopped by user")
    except Exception as e:
        logger.error(f"Error running chat interface: {e}", exc_info=True)
        sys.exit(1)