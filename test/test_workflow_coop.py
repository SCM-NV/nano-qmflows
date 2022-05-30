"""Test the COOP workflow."""

import os
import sys
from pathlib import Path
from os.path import join

import pytest
import numpy as np

from nanoqm.workflows.input_validation import process_input
from nanoqm.workflows.workflow_coop import \
    workflow_crystal_orbital_overlap_population

from .utilsTest import PATH_TEST


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Workflow tends to get stuck and timeout on GitHub Actions for reasons unknown",
)
def test_workflow_coop(tmp_path: Path) -> None:
    """Test the Crystal Orbital Overlap Population workflow."""
    file_path = PATH_TEST / 'input_test_coop.yml'
    config = process_input(file_path, 'coop_calculation')

    # create scratch path
    config.path_hdf5 = join(tmp_path, "HF.hdf5")
    config.workdir = tmp_path

    ref = np.array([
        [-29.309432983398438, 0.07755493190601538],
        [-11.921581268310547, 0.03888611797746827],
        [-8.125808715820312, 5.685036211368225e-20],
        [-8.125808715820312, 1.564841425458713e-21],
        [6.837515830993652, -0.7486973445492844],
    ])
    try:
        coop = workflow_crystal_orbital_overlap_population(config)
        np.testing.assert_allclose(coop, ref, atol=5e-04)

        # Check restart
        coop2 = workflow_crystal_orbital_overlap_population(config)
        np.testing.assert_allclose(coop2, ref, atol=5e-04)
    finally:
        if os.path.isfile("COOP.txt"):
            os.remove("COOP.txt")
