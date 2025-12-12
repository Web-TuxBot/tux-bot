import logging
from datetime import datetime


logger = logging.getLogger("inference_gateway")
logger.setLevel(logging.DEBUG)
log_date = datetime.now().strftime("%Y-%m-%d")

file_handler = logging.FileHandler(f"services/inference_gateway/logs/inference_gateway_{log_date}.log")
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)