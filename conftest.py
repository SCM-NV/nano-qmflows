from __future__ import annotations

import os
import shutil
from collections.abc import Generator

import pytest


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
