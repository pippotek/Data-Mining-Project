# This script sets up and returns a logger instance.
# While WanDB will take care of tracking the metrics and hyperparams
# this script here will help us debugging Spark/Python errors by saving the execution details locally.
# e.g. (db connection issues, problems in the pipeline, etc.)
# The RotatingFileHandler was setup to auto-delete loggings previous to the last 5 runs. This parameter is adjustable
# in the get_logger function below.

import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str, log_file: str = None, level: int = logging.INFO, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True) #Create directory if it doesn't exist
            
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
