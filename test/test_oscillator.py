from nac.common import retrieve_hdf5_data
from nac.workflows.input_validation import process_input
from nac.workflows.workflow_absorption_spectrum import workflow_oscillator_strength
from os.path import join
from .utilsTest import copy_basis_and_orbitals

import numpy as np
import pkg_resources as pkg
import os
import shutil
import tempfile


# Environment data
file_path = pkg.resource_filename('nac', '')
root = os.path.split(file_path)[0]

path_traj_xyz = join(root, 'test/test_files/Cd.xyz')
path_original_hdf5 = join(root, 'test/test_files/Cd.hdf5')
project_name = 'Cd'
input_file = join(root, 'test/test_files/input_test_oscillator.yml')


def test_oscillators_multiprocessing():
    """
    test the oscillator strenght computation using the
    multiprocessing module
    """
    compute_oscillators('multiprocessing')


def test_oscillators_mpi():
    compute_oscillators('mpi')


def compute_oscillators(runner):
    """
    Compute the oscillator strenght and check the results.
    """
    scratch_path = join(tempfile.gettempdir(), 'namd')
    path_test_hdf5 = tempfile.mktemp(
        prefix='{}_'.format(runner), suffix='.hdf5', dir=scratch_path)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path, exist_ok=True)
    try:
        # Run the actual test
        copy_basis_and_orbitals(path_original_hdf5, path_test_hdf5,
                                project_name)
        calculate_oscillators(runner, path_test_hdf5, scratch_path)
        check_properties(path_test_hdf5)

    finally:
        # remove tmp data and clean global config
        shutil.rmtree(scratch_path)


def calculate_oscillators(runner, path_test_hdf5, scratch_path):
    """
    Compute a couple of couplings with the Levine algorithm
    using precalculated MOs.
    """
    config = process_input(input_file, 'absorption_spectrum')
    config['general_settings']['path_hdf5'] = path_test_hdf5
    config['work_dir'] = scratch_path
    config['general_settings']['path_traj_xyz'] = join(
        root, config['general_settings']['path_traj_xyz'])
    config['general_settings']['runner'] = runner
    print(config)

    workflow_oscillator_strength(config)


def check_properties(path_test_hdf5):
    """
    Check that the tensor stored in the HDF5 are correct.
    """
    dipole_matrices = retrieve_hdf5_data(
        path_test_hdf5, 'Cd/multipole/point_0/dipole')

    # The diagonals of each component of the matrix must be zero
    # for a single atom
    diagonals = np.sum([np.diag(dipole_matrices[n]) for n in range(3)])
    assert abs(diagonals) < 1e-16


if __name__ == "__main__":
    test_oscillators_multiprocessing()
