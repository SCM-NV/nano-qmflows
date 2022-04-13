"""The Nano-QMFlows logger."""

from __future__ import annotations

import os
import sys
import types
import logging
import contextlib
from typing import ClassVar

from qmflows.type_hints import PathLike

__all__ = ["logger", "stdout_handler", "EnableFileHandler"]

#: The Nano-QMFlows logger.
logger = logging.getLogger("nanoqm")
logger.setLevel(logging.DEBUG)

qmflows_logger = logging.getLogger("qmflows")
noodles_logger = logging.getLogger("noodles")
noodles_logger.setLevel(logging.WARNING)

#: The Nano-QMFlows stdout handler.
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logging.Formatter(
    fmt='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
))
logger.addHandler(stdout_handler)


class EnableFileHandler(contextlib.ContextDecorator):
    """Add a file handler to the noodles, qmflows and nanoqm loggers.

    Attributes
    ----------
    handler : logging.FileHandler
        The relevant titular handler.

    """

    __slots__ = ("handler",)

    LOGGERS: ClassVar = (logger, qmflows_logger, noodles_logger)

    def __init__(self, path: PathLike) -> None:
        """Initialize the context manager.

        Parameters
        ----------
        path : path-like object
            Path to the log file.

        """
        self.handler = logging.FileHandler(os.fsdecode(path))
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s---%(levelname)s\n%(message)s\n',
            datefmt='%H:%M:%S',
        ))

    def __enter__(self) -> None:
        """Add the file handler."""
        for logger in self.LOGGERS:
            if self.handler not in logger.handlers:
                logger.addHandler(self.handler)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: types.TracebackType | None,
    ) -> None:
        """Remove the file handler."""
        for logger in self.LOGGERS:
            if self.handler in logger.handlers:
                logger.removeHandler(self.handler)
