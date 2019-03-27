from .utilsTest import (cp2k_available, remove_files)
from nac.common import search_data_in_hdf5
from nac.workflows.input_validation import process_input
from nac.workflows.workflow_single_points import workflow_single_points
from os.path import join

import pkg_resources as pkg
import pytest
import os
import sys


# Environment data
file_path = pkg.resource_filename('nac', '')
root = os.path.split(file_path)[0]


@pytest.mark.skipif(
    not cp2k_available(), reason="CP2K is not install or not loaded")
def test_single_point(tmp_path):
    """
    Check that the couplings run
    """
    file_path = join(root, 'test/test_files/input_test_single_points.yml')
    config = process_input(file_path, 'single_points')
    print("config: ", config)
    config["scratch_path"] = tmp_path
    tmp_hdf5 = os.path.join(tmp_path, 'single_points.hdf5')
    config['path_hdf5'] = tmp_hdf5
    try:
        path_orbitals = workflow_single_points(config)
        print("path_orbitals: ", path_orbitals)
        check_orbitals(path_orbitals[0], tmp_hdf5)
    except:
        print("scratch_path: ", tmp_path)
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        remove_files


def check_orbitals(orbitals: list, path_hdf5: str) -> None:
    """
    Check that the orbitals are stored in the HDF5
    """
    assert search_data_in_hdf5(path_hdf5, orbitals)
