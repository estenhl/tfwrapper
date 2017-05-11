import logging

class Logger():
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    def __init__(self):
        self.logger = logging.getLogger('tfwrapper')

        self.ch = logging.StreamHandler()
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s', '%H:%M:%S')
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        self.setLevel(logging.INFO)

    def setLevel(self, level):
        self.logger.setLevel(level)
        self.ch.setLevel(level)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

logger = Logger()