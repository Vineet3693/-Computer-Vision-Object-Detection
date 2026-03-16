"""
Logging utility for the Object Detection System
Handles logging to file and console
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup and return a logger instance.
    
    Args:
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger('ObjectDetection', 'outputs/logs/app.log')


def log_detection(detection_data: dict):
    """
    Log detection results.
    
    Args:
        detection_data: Dictionary containing detection information
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log_entry = {
        'timestamp': timestamp,
        'input_type': detection_data.get('input_type', 'unknown'),
        'total_detections': detection_data.get('total_detections', 0),
        'person_count': detection_data.get('person_count', 0),
        'vehicle_count': detection_data.get('vehicle_count', 0),
        'avg_confidence': detection_data.get('avg_confidence', 0.0),
        'inference_time_ms': detection_data.get('inference_time_ms', 0.0),
        'fps': detection_data.get('fps', 0.0),
    }
    
    logger.info(f"Detection: {log_entry}")
    return log_entry


def log_error(error_message: str, exception: Exception = None):
    """
    Log error messages.
    
    Args:
        error_message: Error message string
        exception: Optional exception object
    """
    if exception:
        logger.error(f"{error_message}: {str(exception)}", exc_info=True)
    else:
        logger.error(error_message)


def log_info(message: str):
    """
    Log info messages.
    
    Args:
        message: Info message string
    """
    logger.info(message)


def log_warning(message: str):
    """
    Log warning messages.
    
    Args:
        message: Warning message string
    """
    logger.warning(message)
