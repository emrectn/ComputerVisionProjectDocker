import sys
import logging
from logging import DEBUG, INFO, ERROR
from logging.handlers import RotatingFileHandler

class MyLogger(object):
    def __init__(self, name, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", level=INFO):
        # Initial construct.
        self.format = format
        self.level = level
        self.name = name

        # Logger configuration.
        self.console_formatter = logging.Formatter(self.format)
        self.console_logger = RotatingFileHandler('logs/python.log', maxBytes=1024 * 1024 * 100, backupCount=10)
        self.console_logger.setFormatter(self.console_formatter)

        # Complete logging config.
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.addHandler(self.console_logger)

    def info(self, msg, extra=None):
        self.logger.info(msg, extra=extra)

    def error(self, msg, extra=None,exc_info=True):
        self.logger.error(msg, extra=extra,exc_info=True)

    def debug(self, msg, extra=None):
        self.logger.debug(msg, extra=extra)

    def warn(self, msg, extra=None):
        self.logger.warn(msg, extra=extra)