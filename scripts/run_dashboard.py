#!/usr/bin/env python
"""
Run the RAG pipeline monitoring dashboard.
"""
import streamlit.web.bootstrap as bootstrap
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragstack.web.dashboard import create_progress_dashboard
from ragstack.utils.logging import setup_logger

# Initialize logger
logger = setup_logger("run_dashboard")

def main():
    """Run the dashboard."""
    logger.info("Starting RAG pipeline monitoring dashboard")
    
    # Set the script path to the dashboard module
    dashboard_path = project_root / "ragstack/web/dashboard.py"
    
    # Tell streamlit where to find the script
    os.environ["STREAMLIT_APP"] = str(dashboard_path)
    
    # Run the dashboard
    # Pass empty list as args, not None
    bootstrap.run(dashboard_path, [], {}, None)

if __name__ == "__main__":
    main()