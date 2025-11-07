import logging
import sys

def get_logger(name=None):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(fmt="[%(asctime)s] %(levelname)s %(funcName)s: %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "CRITICAL": "\x1b[95m",  # bright magenta
        "ERROR":    "\x1b[91m",  # red
        "WARNING":  "\x1b[93m",  # yellow
        "INFO":     "\x1b[92m",  # green
        "DEBUG":    "\x1b[36m",  # cyan
    }
    RESET = "\x1b[0m"

    def __init__(self, fmt=None, datefmt=None, style="%", use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record):
        message = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelname, "")
            return f"{color}{message}{self.RESET}"
        return message
