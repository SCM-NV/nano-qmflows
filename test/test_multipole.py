"""Test quadrupole calculation."""

import shutil
from pathlib import Path

import numpy as np
from assertionlib import assertion
from nanoqm.integrals.multipole_matrices import compute_matrix_multipole
from nanoqm.workflows.input_validation import process_input
from qmflows.parsers.xyzParser import readXYZ

from .utilsTest import PATH_TEST


def test_quadropole(tmp_path):
    """Test the calculation of a quadrupole."""
    file_path = PATH_TEST / "input_test_single_points.yml"
    config = process_input(file_path, 'single_points')
    path_original_hdf5 = config.path_hdf5
    path_test_hdf5 = (Path(tmp_path) / "multipoles.hdf5").as_posix()

    # copy the precomputed data to the temporal HDF5
    shutil.copyfile(path_original_hdf5, path_test_hdf5)
    config.path_hdf5 = path_test_hdf5

    mol = readXYZ((PATH_TEST / "ethylene.xyz").as_posix())
    matrix = compute_matrix_multipole(mol, config, "quadrupole")
    # The matrix contains the overlap + dipole + quadrupole
    assertion.shape_eq(matrix, (10, 46, 46))
    # Check that the matrices are symmetric
    for i in range(10):
        arr = matrix[i].reshape(46, 46)
        assertion.truth(np.allclose(arr, arr.T))
