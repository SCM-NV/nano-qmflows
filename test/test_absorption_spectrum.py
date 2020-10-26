"""Test he absorption spectrum workflows."""
import shutil
from os.path import join
from pathlib import Path

import numpy as np
from nanoqm.common import retrieve_hdf5_data
from nanoqm.workflows import workflow_stddft
from nanoqm.workflows.input_validation import process_input

from .utilsTest import PATH_TEST, remove_files


def run_and_check(tmp_path: Path, project: str, input_file: str, orbitals_type: str = ""):
    """Run the workflow and check the results."""
    path_original_hdf5 = PATH_TEST / f'{project}.hdf5'
    # create scratch path
    shutil.copy(path_original_hdf5, tmp_path)
    for approx in ("sing_orb", "stda"):
        try:
            # Run the actual test
            path_test_hdf5 = Path(tmp_path) / f"{project}_{approx}.hdf5"
            shutil.copyfile(path_original_hdf5, path_test_hdf5)
            calculate_oscillators(path_test_hdf5, tmp_path, approx, input_file)
            check_properties(path_test_hdf5, orbitals_type)

            # Run again the workflow to check that the data is read from the hdf5
            calculate_oscillators(path_test_hdf5, tmp_path, approx, input_file)
            check_properties(path_test_hdf5, orbitals_type)
        finally:
            remove_files()


def test_compute_oscillators(tmp_path: Path):
    """Compute the oscillator strenght and check the results."""
    # path_original_hdf5 = PATH_TEST / 'Cd.hdf5'
    input_file = 'input_test_absorption_spectrum.yml'
    run_and_check(tmp_path, "Cd", input_file)


def test_compute_oscillators_unrestricted(tmp_path: Path):
    """Compute the oscillator strenght and check the results."""
    input_file = 'input_test_absorption_spectrum_unrestricted.yml'
    run_and_check(tmp_path, "oxygen", input_file, "alphas")


def calculate_oscillators(path_test_hdf5: Path, scratch_path: Path, approx: str, file_name: str):
    """Compute a couple of couplings with the Levine algorithm using precalculated MOs."""
    input_file = PATH_TEST / file_name
    config = process_input(input_file, 'absorption_spectrum')
    config['path_hdf5'] = path_test_hdf5.absolute().as_posix()
    config['scratch_path'] = scratch_path
    config['workdir'] = scratch_path
    config['tddft'] = approx
    config['path_traj_xyz'] = Path(config.path_traj_xyz).absolute().as_posix()

    workflow_stddft(config)


def check_properties(path_test_hdf5: Path, orbitals_type: str):
    """Check that the tensor stored in the HDF5 are correct."""
    path_dipole = join(orbitals_type, 'dipole', 'point_0')
    dipole_matrices = retrieve_hdf5_data(
        path_test_hdf5, path_dipole)

    # The diagonals of each component of the matrix must be zero
    # for a single atom
    diagonals = np.sum([np.diag(dipole_matrices[n + 1]) for n in range(3)])
    assert abs(diagonals) < 1e-16
