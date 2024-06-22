import logging
from .log import set_logger

logger = set_logger(logging.INFO)
logging.info(f'setting logging config at {__path__}')
