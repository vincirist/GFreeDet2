import logging as logging_ori


import traceback
from typing import Union

import numpy as np

FORMAT_PREFIX = "%(levelname).1s%(asctime)s.%(msecs)d %(process)d %(filename)s:%(lineno)d]"
FORMAT = f"{FORMAT_PREFIX} %(message)s"
DATEFMT = "%m%d %H:%M:%S"


def _seq_repr(parts, prefix, suffix, indent):
    if len(parts) <= 1 or sum(len(p) for p in parts) < 80:
        return prefix + ", ".join(parts) + suffix
    return f"{prefix}\n{indent}  " + f",\n{indent}  ".join(parts) + f"\n{indent}{suffix}"


class LocalsFormatter(logging_ori.Formatter):

    def formatException(self, exc_info) -> str:
        tb = traceback.TracebackException(*exc_info, capture_locals=True)
        # try not to be too verbose
        for frame in tb.stack:
            if len(repr(frame.locals)) > 500:
                frame.locals = None
        return "".join(tb.format())


def config_logging(
    *,
    fmt: str = FORMAT,
    level: Union[int, str] = logging_ori.INFO,
    datefmt: str = DATEFMT,
    style: str = "%",
    stream=None,
) -> None:

    root = logging_ori.getLogger()

    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    h = logging_ori.StreamHandler(stream)
    h.setFormatter(LocalsFormatter(fmt, datefmt, style))
    root.addHandler(h)
    root.setLevel(level)


DEBUG: int = logging_ori.DEBUG
INFO: int = logging_ori.INFO
WARNING: int = logging_ori.WARNING
ERROR: int = logging_ori.ERROR

WHITE = "\x1b[37;20m"
WHITE_BOLD = "\x1b[37;1m"
BLUE = "\x1b[34;20m"
BLUE_BOLD = "\x1b[34;1m"
RED = "\x1b[31;20m"
RED_BOLD = "\x1b[31;1m"
RESET = "\x1b[0m"

Logger = logging_ori.Logger


def get_logger(level: int = logging_ori.INFO) -> Logger:

    config_logging(level=level)
    return logging_ori.getLogger(__name__)


def get_logger_v2(name: str, level: int = logging_ori.INFO):
    logger = logging_ori.getLogger(name)
    logger.setLevel(level)
    return logger


def get_separator(length: int = 80) -> str:

    return length * "-"


def log_heading(logger: Logger, msg: str, style: str = WHITE) -> None:

    separator = get_separator()
    logger.info(style + separator + RESET)
    logger.info(style + msg + RESET)
    logger.info(style + separator + RESET)
