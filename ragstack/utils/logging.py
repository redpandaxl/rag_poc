"""
Logging utilities for the RAG application.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from ragstack.config.settings import settings


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with file and/or console handlers.
    
    Args:
        name: The name of the logger
        log_file: Optional file path for logging
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Use default settings if not specified
    if log_file is None:
        log_file = settings.log_file
    
    if level is None:
        # Check for DEBUG environment variable to enable detailed logging
        if os.environ.get("DEBUG_LOGGING", "0") == "1":
            level = "DEBUG"
        else:
            level = settings.log_level
    
    # Convert string level to logging level
    log_level = getattr(logging, level)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    standard_formatter = logging.Formatter(settings.log_format)
    
    # More detailed formatter for debugging with timestamps, module, function, and line number
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
    )
    
    # Choose formatter based on log level
    formatter = detailed_formatter if level == "DEBUG" else standard_formatter
    
    # Processing log file - separate from main log file for data processing
    if name.startswith("ragstack.data_ingestion") or name == "run_ingestion":
        processing_log_file = settings.logs_dir / "processing.log"
        processing_log_file.parent.mkdir(parents=True, exist_ok=True)
        processing_handler = logging.FileHandler(processing_log_file)
        processing_handler.setLevel(log_level)
        processing_handler.setFormatter(formatter)
        logger.addHandler(processing_handler)
    
    # File handler for main log
    if log_file:
        # Create directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Log logger creation
    logger.debug(f"Logger {name} initialized with level {level}")
    
    return logger


# Create a default logger for the application
app_logger = setup_logger("ragstack")
