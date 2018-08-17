from functools import partial
from nac.common import retrieve_hdf5_data
from nac.workflows.input_validation import process_input
from nac.workflows.workflow_coupling import workflow_derivative_couplings
from os.path import join
from .utilsTest import copy_basis_and_orbitals

import numpy as np
import pkg_resources as pkg
import pytest
import os
import shutil
import tempfile


# Environment data
file_path = pkg.resource_filename('nac', '')
root = os.path.split(file_path)[0]

path_traj_xyz = join(root, 'test/test_files/Cd33Se33_fivePoints.xyz')
path_original_hdf5 = join(root, 'test/test_files/Cd33Se33.hdf5')
project_name = 'Cd33Se33'
input_file = join(root, 'test/test_files/input_test_derivative_couplings.yml')


@pytest.mark.slow
def test_couplings_multiprocessing():
    """
    Test couplings calculations for Cd33Se33
    """
    compute_derivative_coupling('multiprocessing')


@pytest.mark.slow
def test_couplings_mpi():
    """
    Test couplings calculations for Cd33Se33
    """
    compute_derivative_coupling('mpi')


def compute_derivative_coupling(runner):
    scratch_path = join(tempfile.gettempdir(), 'namd')
    path_test_hdf5 = tempfile.mktemp(
        prefix='{}_'.format(runner), suffix='.hdf5', dir=scratch_path)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)
    try:
        # Run the actual test
        copy_basis_and_orbitals(path_original_hdf5, path_test_hdf5,
                                project_name)
        calculate_couplings(runner, path_test_hdf5, scratch_path)
        check_properties(path_test_hdf5)
    finally:
        shutil.rmtree(scratch_path)


def calculate_couplings(runner, path_test_hdf5, scratch_path):
    """
    Compute some of couplings with the Levine algorithm
    using precalculated MOs.
    """
    config = process_input(input_file, 'derivative_couplings')
    config['general_settings']['path_hdf5'] = path_test_hdf5
    config['work_dir'] = scratch_path
    config['general_settings']['path_traj_xyz'] = join(
        root, config['general_settings']['path_traj_xyz'])
    config['general_settings']['runner'] = runner
    workflow_derivative_couplings(config)


def check_properties(path_test_hdf5):
    """
    Test if the coupling coupling by the Levine method is correct
    """
    # Paths to all the arrays to test
    path_swaps = join(project_name, 'swaps')
    name_Sji = 'overlaps_{}/mtx_sji_t0'
    name_Sji_fixed = 'overlaps_{}/mtx_sji_t0_corrected'
    path_overlaps = [join(project_name, name_Sji.format(i)) for i in range(4)]
    path_fixed_overlaps = [join(project_name, name_Sji_fixed.format(i))
                           for i in range(4)]
    path_couplings = [join(project_name, 'coupling_{}'.format(i))
                      for i in range(4)]

    # Define partial func
    fun_original = partial(stack_retrieve, path_original_hdf5)
    fun_test = partial(stack_retrieve, path_test_hdf5)

    # Read data from the HDF5
    swaps_original = retrieve_hdf5_data(path_original_hdf5, path_swaps)
    swaps_test = retrieve_hdf5_data(path_test_hdf5, path_swaps)

    overlaps_original = fun_original(path_overlaps)
    overlaps_test = fun_test(path_overlaps)

    fixed_overlaps_original = fun_original(path_fixed_overlaps)
    fixed_overlaps_test = fun_test(path_fixed_overlaps)

    css_original = fun_original(path_couplings)
    css_test = fun_test(path_couplings)

    # Test data
    b1 = np.allclose(swaps_original, swaps_test)
    b2 = np.allclose(overlaps_original, overlaps_test)
    b3 = np.allclose(fixed_overlaps_original, fixed_overlaps_test)
    b4 = np.allclose(css_original, css_test)

    assert all((b1, b2, b3, b4))


def stack_retrieve(path_hdf5, path_prop):
    """
    Retrieve a list of Numpy arrays and create a tensor out of it
    """
    return np.stack(retrieve_hdf5_data(path_hdf5, path_prop))
