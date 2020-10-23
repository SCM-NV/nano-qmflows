"""Test he absorption spectrum workflows."""
import shutil
from pathlib import Path

import numpy as np
from qmflows.type_hints import PathLike

from nanoqm.common import retrieve_hdf5_data
from nanoqm.workflows import workflow_stddft
from nanoqm.workflows.input_validation import process_input

from .utilsTest import PATH_TEST, copy_basis_and_orbitals, remove_files


def test_compute_oscillators(tmp_path):
    """Compute the oscillator strenght and check the results."""
    project_name = 'Cd'
    path_original_hdf5 = PATH_TEST / 'Cd.hdf5'

    # create scratch path
    shutil.copy(path_original_hdf5, tmp_path)
    for approx in ("sing_orb", "stda"):
        try:
            # Run the actual test
            path_test_hdf5 = Path(tmp_path) / f"Cd_{approx}.hdf5"
            copy_basis_and_orbitals(path_original_hdf5, path_test_hdf5,
                                    project_name)
            calculate_oscillators(path_test_hdf5, tmp_path, approx)
            check_properties(path_test_hdf5)
        finally:
            remove_files()


def calculate_oscillators(path_test_hdf5: Path, scratch_path: PathLike, approx: str):
    """Compute a couple of couplings with the Levine algorithm using precalculated MOs."""
    input_file = PATH_TEST / 'input_test_absorption_spectrum.yml'
    config = process_input(input_file, 'absorption_spectrum')
    config['path_hdf5'] = path_test_hdf5.absolute().as_posix()
    config['scratch_path'] = scratch_path
    config['workdir'] = scratch_path
    config['tddft'] = approx
    config['path_traj_xyz'] = Path(config.path_traj_xyz).absolute().as_posix()

    workflow_stddft(config)


def check_properties(path_test_hdf5):
    """Check that the tensor stored in the HDF5 are correct."""
    dipole_matrices = retrieve_hdf5_data(
        path_test_hdf5, 'Cd/multipole/point_0/dipole')

    # The diagonals of each component of the matrix must be zero
    # for a single atom
    diagonals = np.sum([np.diag(dipole_matrices[n + 1]) for n in range(3)])
    assert abs(diagonals) < 1e-16
