from nac.common import retrieve_hdf5_data
from nac.workflows.input_validation import process_input
from nac.workflows.workflow_stddft_spectrum import workflow_stddft
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

path_original_hdf5 = join(root, 'test/test_files/ethylene.hdf5')
project_name = 'ethylene'
input_file = join(root, 'test/test_files/input_test_tddft.yml')


def test_oscillators_multiprocessing():
    """
    test the oscillator strenght computation using the
    multiprocessing module
    """
    compute_stdft('multiprocessing')


def compute_stdft(runner):
    """
    Compute the oscillator strenght and check the results.
    """
    scratch_path = join(tempfile.gettempdir(), 'namd')
    path_test_hdf5 = tempfile.mktemp(
        prefix='{}_'.format(runner), suffix='.hdf5', dir=scratch_path)
    print(path_test_hdf5)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)
    try:
        # Run the actual test
        copy_basis_and_orbitals(path_original_hdf5, path_test_hdf5,
                                project_name)
        results = calculate_stdft(runner, path_test_hdf5, scratch_path)
        check_properties(results)

    finally:
        # remove tmp data and clean global config
        # shutil.rmtree(scratch_path)
        pass


def calculate_stdft(runner, path_test_hdf5, scratch_path):
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
    return workflow_stddft(config)


def check_properties(results):
    """
    Check that the tensor stored in the HDF5 are correct.
    """
    matrices = [np.loadtxt(path) for path in results]
    sum_oscillators = np.sum(matrices[0][:, 2])

    assert sum_oscillators < 8 and sum_oscillators > 0


if __name__ == "__main__":
    test_oscillators_multiprocessing()
