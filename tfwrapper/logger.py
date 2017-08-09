import inspect
import logging

class Logger():
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    def __init__(self):
        self.logger = logging.getLogger('tfwrapper')

        self.ch = logging.StreamHandler()
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%H:%M:%S')
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        self.setLevel(self.DEBUG)

    def log(self, lvl, msg):
        frame = inspect.stack()[1]

        # If the call comes from one of Loggers functions (debug, warning ...), use the caller before that on the stack
        if frame.filename.endswith('/logger.py'):
            frame = inspect.stack()[2]

        filename = frame.filename.split('/')[-1]
        lineno = frame.lineno
        caller = frame.function
        self.logger.log(lvl, '%s:%d %s(): %s' % (filename, lineno, caller, msg))

    def setLevel(self, level):
        self.logger.setLevel(level)
        self.ch.setLevel(level)

    def debug(self, msg):
        self.log(self.DEBUG, msg)

    def info(self, msg):
        self.log(self.INFO, msg)

    def warning(self, msg):
        self.log(self.WARNING, msg)

    def error(self, msg):
        self.log(self.ERROR, msg)

logger = Logger()