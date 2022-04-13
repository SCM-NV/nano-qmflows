from __future__ import annotations

import os
import shutil
from collections.abc import Generator

import pytest
from nanoqm._logger import logger, stdout_handler


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
    assert stdout_handler in logger.handlers
    logger.removeHandler(stdout_handler)
    yield None
    logger.addHandler(stdout_handler)
