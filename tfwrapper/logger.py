import logging

logger = logging.getLogger('tfwrapper')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s', '%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)