from __future__ import annotations

import logging

import os
import shutil
from collections.abc import Generator

import pytest
from nanoqm._logger import logger as nanoqm_logger


@pytest.fixture(autouse=True, scope="session")
def is_release() -> Generator[bool, None, None]:
    """Yield whether the test suite is run for a nano-qmflows release or not."""
    env_var = os.environ.get("IS_RELEASE", 0)
    try:
        yield bool(int(env_var))
    except ValueError as ex:
        raise ValueError("The `IS_RELEASE` environment variable expected an integer") from ex


@pytest.fixture(autouse=True, scope="function")
def cleunup_files() -> Generator[None, None, None]:
    # TODO: Investigate if these files can be removed by their respective test(s)
    yield None
    if os.path.isfile("quantum.hdf5"):
        os.remove("quantum.hdf5")
    if os.path.isfile("input_parameters.yml"):
        os.remove("input_parameters.yml")
    if os.path.isdir("overlaps"):
        shutil.rmtree("overlaps")


@pytest.fixture(autouse=True, scope="session")
def prepare_logger() -> Generator[None, None, None]:
    """Remove the logging output to stdout while running tests."""
    import noodles
    import qmflows
    noodles_logger = logging.getLogger("noodles")
    qmflows_logger = logging.getLogger("qmflows")

    nanoqm_handlers = nanoqm_logger.handlers.copy()
    noodles_handlers = noodles_logger.handlers.copy()
    qmflows_handlers = qmflows_logger.handlers.copy()

    for handler in nanoqm_handlers:
        nanoqm_logger.removeHandler(handler)
    for handler in noodles_handlers:
        noodles_logger.removeHandler(handler)
    for handler in qmflows_handlers:
        qmflows_logger.removeHandler(handler)

    yield None

    for handler in nanoqm_handlers:
        nanoqm_logger.addHandler(handler)
    for handler in noodles_handlers:
        noodles_logger.addHandler(handler)
    for handler in qmflows_handlers:
        qmflows_logger.addHandler(handler)
