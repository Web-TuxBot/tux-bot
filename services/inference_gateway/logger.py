import logging
import colorlog
from datetime import datetime


logger = logging.getLogger("inference_gateway")
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s %(levelname)s%(reset)s - %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red,bg_white'
    },
    datefmt='%Y-%m-%d %H:%M:%C',
    reset=True,
    style='%' 
    )

stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)