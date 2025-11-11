import logging
import sys
import os
from pathlib import Path

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

class TensorboardLogger():
    def __init__(self, name=None):
        self.path = Path(self.log_dir) / name
        self.writer = self.create_tensorboard_writer(log_dir=self.path)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_loss(self, values, step):
        self.writer.add_scalars("Loss", values, global_step=step)

    def log_lr(self, value, step):
        self.writer.add_scalar("Learning_Rate", value, step)

    def log_precision(self, values, step):
        acc = {k : v["precision"] for k, v in values.items()}
        self.writer.add_scalars("Precision", acc, step)

    def log_f1(self, values, step):
        acc = {k : v["f1"] for k, v in values.items()}
        self.writer.add_scalars("F1-Score", acc, step)

    def log_sensitivity(self, values, step):
        acc = {k: v["sensitivity"] for k, v in values.items()}
        self.writer.add_scalars("Recall", acc, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, step)


    @property
    def log_dir(self):
        return os.environ["TENSORBOARD_DIR"]


    def create_tensorboard_writer(self, config=None, log_dir=None):
        # TODO: THINK about location of writer? Should it be inside the hashed dir?
        # from data.cache import DataCacher
        from torch.utils.tensorboard import SummaryWriter

        # if config is not None:
        #     cacher = DataCacher(config)
        #     if log_dir is None:
        #         log_dir = cacher.cache_dir()
        return SummaryWriter(log_dir=log_dir)
