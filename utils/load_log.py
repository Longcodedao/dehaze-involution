import logging

def setup_logger(logger_name, log_file, level=logging.INFO):
    """Function to setup logger"""

    logger = logging.getLogger(logger_name)
    # Create a custom logger
    handler = logging.FileHandler(log_file)    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

