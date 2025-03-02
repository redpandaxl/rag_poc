#!/usr/bin/env python
"""
Run the test suite for the RAG application.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragstack.utils.logging import setup_logger

# Initialize logger
logger = setup_logger("run_tests")

def main():
    """Run the test suite."""
    parser = argparse.ArgumentParser(description="Run RAG pipeline tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-path", help="Specific test path to run")
    parser.add_argument("--xvs", action="store_true", help="Show extra verbose output")
    args = parser.parse_args()
    
    logger.info("Starting test suite")
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        logger.error("pytest not found. Please install it with: pip install pytest")
        return 1
    
    # Build the pytest command
    pytest_args = ["-xvs"] if args.xvs else ["-v"] if args.verbose else []
    
    if args.test_path:
        test_path = args.test_path
        logger.info(f"Running tests in: {test_path}")
    else:
        test_path = str(project_root / "ragstack" / "tests")
        logger.info(f"Running all tests in: {test_path}")
    
    pytest_args.append(test_path)
    
    # Run pytest
    logger.info(f"Running pytest with args: {pytest_args}")
    result = pytest.main(pytest_args)
    
    if result == 0:
        logger.info("All tests passed!")
    else:
        logger.error(f"Tests failed with code: {result}")
    
    return result

if __name__ == "__main__":
    sys.exit(main())