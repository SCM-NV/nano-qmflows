"""Test the COOP workflow."""
import os
import shutil
import sys
from os.path import join

import pkg_resources as pkg

from nac.workflows.input_validation import process_input
from nac.workflows.workflow_coop import \
    workflow_crystal_orbital_overlap_population

# Environment data
file_path = pkg.resource_filename('nac', '')
root = os.path.split(file_path)[0]


def test_workflow_coop(tmp_path):
    """Test the Crystal Orbital Overlap Population workflow."""
    file_path = join(root, 'test/test_files/input_test_coop.yml')
    config = process_input(file_path, 'coop_calculation')

    # create scratch path
    shutil.copy(config.path_hdf5, tmp_path)
    config.path_hdf5 = join(tmp_path, "Cd33Se33.hdf5")
    config.workdir = tmp_path
    try:
        workflow_crystal_orbital_overlap_population(config)
        os.remove("COOP.txt")
    except BaseException:
        print("scratch_path: ", tmp_path)
        print("Unexpected error:", sys.exc_info()[0])
        raise
