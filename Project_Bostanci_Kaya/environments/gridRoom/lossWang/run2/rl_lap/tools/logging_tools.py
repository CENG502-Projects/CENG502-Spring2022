import logging
import os
import sys
import datetime
import re


def config_logging(log_dir, filename='log.txt'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filepath = os.path.join(log_dir, filename)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath, 'w')
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    logger.addHandler(ch)


def get_datetime():
    now = datetime.datetime.now().isoformat()
    now = re.sub(r'\D', '', now)[:-6]
    return now