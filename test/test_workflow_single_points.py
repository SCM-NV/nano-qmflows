"""Test a single point calculation using CP2K."""
import os
from pathlib import Path

import pytest
from assertionlib import assertion

from nanoqm.common import is_data_in_hdf5
from nanoqm.workflows.input_validation import process_input
from nanoqm.workflows.workflow_single_points import workflow_single_points

from .utilsTest import PATH_TEST, cp2k_available, remove_files


def run_single_point(tmp_path: Path, input_file: str):
    """Run a single point calculation using cp2k."""
    file_path = PATH_TEST / input_file
    config = process_input(file_path, 'single_points')
    config["scratch_path"] = tmp_path
    tmp_hdf5 = os.path.join(tmp_path, 'single_points.hdf5')
    config['path_hdf5'] = tmp_hdf5
    Path(tmp_hdf5).touch()
    try:
        path_orbitals, path_energies = workflow_single_points(config)
        print("path_orbitals: ", path_orbitals)
        print("path_energies: ", path_energies)
        if config["compute_orbitals"]:
            assertion.truth(is_data_in_hdf5(tmp_hdf5, path_orbitals[0]))
    finally:
        remove_files()


# @pytest.mark.skipif(
#     not cp2k_available(), reason="CP2K is not install or not loaded")
@pytest.mark.slow
def test_single_point(tmp_path: Path):
    """Check that the couplings run."""
    run_single_point(tmp_path, "input_test_single_points.yml")
