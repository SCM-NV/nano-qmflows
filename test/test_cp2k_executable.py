"""Test the handling of the CP2K executable."""
import os
from pathlib import Path

import pytest

from nac.workflows.input_validation import process_input
from nac.workflows.workflow_single_points import workflow_single_points
from qmflows.type_hints import PathLike

from .utilsTest import PATH_TEST, cp2k_available, remove_files


@pytest.mark.skipif(
    not cp2k_available(), reason="CP2K is not install or not loaded")
def test_cp2k_executable(tmp_path: PathLike) -> None:
    """Test CP2K executables other than cp2k.popt."""
    file_path = PATH_TEST / 'input_test_single_points.yml'
    config = process_input(file_path, 'single_points')
    # tmp files
    tmp_hdf5 = os.path.join(tmp_path, 'cp2k_executable.hdf5')
    Path(tmp_hdf5).touch()
    config.path_hdf5 = tmp_hdf5
    config.scratch_path = tmp_path
    config.workdir = tmp_path

    # executable to test
    config.executable = "cp2k.popt"
    try:
        workflow_single_points(config)
    finally:
        remove_files()
