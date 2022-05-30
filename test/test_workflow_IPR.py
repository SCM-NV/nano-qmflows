"""Test the IPR workflow."""

import os
import sys
from os.path import join
from pathlib import Path

import pytest
import numpy as np
from nanoqm.workflows.input_validation import process_input
from nanoqm.workflows.workflow_ipr import workflow_ipr

from .utilsTest import PATH_TEST


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Workflow tends to get stuck and timeout on GitHub Actions for reasons unknown",
)
def test_workflow_IPR(tmp_path: Path) -> None:
    """Test the Inverse Participation Ratio workflow."""
    file_path = PATH_TEST / 'input_test_IPR.yml'
    config = process_input(file_path, 'ipr_calculation')

    # create scratch path
    config.path_hdf5 = join(tmp_path, "F2.hdf5")
    config.workdir = tmp_path

    ref = np.array([
        [-14.944306373596191, 1.4802774436230284],
        [-12.682127952575684, 1.9999967522621063],
        [-12.682127952575684, 1.9999969940206397],
        [-9.349729537963867, 1.9999969936606203],
        [-9.349729537963867, 1.999997231558098],
        [-5.862362384796143, 1.7524837324869675],
    ])
    try:
        ipr = workflow_ipr(config)
        np.testing.assert_allclose(ipr, ref, rtol=5e-04)

        # Check restart
        ipr2 = workflow_ipr(config)
        np.testing.assert_allclose(ipr2, ref, rtol=5e-04)
    finally:
        if os.path.isfile("IPR.txt"):
            os.remove("IPR.txt")
