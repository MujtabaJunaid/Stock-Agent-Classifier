import logging
import os
from datetime import datetime

# Configure logger
logger = logging.getLogger("StockPredictionPipeline")
logger.setLevel(logging.DEBUG)

# Create file handler for logging to a file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create console handler for logging to stdout
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter with line number and file name for detailed error logging
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_logger():
    """Return the configured logger."""
    return logger
