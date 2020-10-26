"""Test the derivative coupling calculation."""
import os
import shutil
from typing import Sequence

import numpy as np

from assertionlib import assertion
from nanoqm.common import DictConfig, is_data_in_hdf5, retrieve_hdf5_data
from nanoqm.workflows.input_validation import process_input
from nanoqm.workflows.workflow_coupling import workflow_derivative_couplings

from .utilsTest import PATH_TEST, remove_files


def test_fast_couplings(tmp_path):
    """Check the derivative couplings workflow"""
    run_derivative_coupling(tmp_path, 'input_fast_test_derivative_couplings.yml')


def test_unrestricted_alphas(tmp_path):
    """Test the derivative coupling for the alphas spin orbitals."""
    run_derivative_coupling(tmp_path, 'input_couplings_alphas.yml', "alphas")


def test_unrestricted_betas(tmp_path):
    """Test the derivative coupling for the alphas spin orbitals."""
    run_derivative_coupling(tmp_path, 'input_couplings_both.yml', "both")


def run_derivative_coupling(tmp_path: str, input_file: str, orbitals_type: str = "") -> None:
    """Check that the couplings run."""
    path_input = PATH_TEST / input_file
    config = process_input(path_input, 'derivative_couplings')
    config["scratch_path"] = tmp_path
    tmp_hdf5 = os.path.join(tmp_path, 'fast_couplings.hdf5')
    shutil.copy(config.path_hdf5, tmp_hdf5)
    config['path_hdf5'] = tmp_hdf5
    config['write_overlaps'] = True
    try:
        check_results(config, tmp_hdf5, orbitals_type)
        # Run the calculation again to test that the data is read from the hdf5
        check_results(config, tmp_hdf5, orbitals_type)
    finally:
        remove_files()


def check_results(config: DictConfig, tmp_hdf5: str, orbitals_type: str) -> None:
    """Check the computed results stored in the HDF5 file."""
    if orbitals_type != "both":
        hamiltonians, _ = workflow_derivative_couplings(config)
        check_couplings(config, tmp_hdf5, orbitals_type)
        check_hamiltonians(hamiltonians)
    else:
        result_alphas, result_betas = workflow_derivative_couplings(config)
        check_couplings(config, tmp_hdf5, "alphas")
        check_couplings(config, tmp_hdf5, "betas")
        check_hamiltonians(result_alphas[0])
        check_hamiltonians(result_betas[0])


def check_couplings(config: DictConfig, tmp_hdf5: str, orbitals_type: str) -> None:
    """Check that the couplings have meaningful values."""
    def create_paths(keyword: str) -> list:
        return [os.path.join(orbitals_type, f'{keyword}_{x}')
                for x in range(len(config.geometries) - 1)]
    overlaps = create_paths('overlaps')
    couplings = create_paths('coupling')

    # Check that couplings and overlaps exists
    assertion.truth(is_data_in_hdf5(tmp_hdf5, overlaps))
    assertion.truth(is_data_in_hdf5(tmp_hdf5, couplings))

    # All the elements are different of inifinity or nan
    tensor_couplings = np.stack(retrieve_hdf5_data(tmp_hdf5, couplings))
    assertion.truth(np.isfinite(tensor_couplings).all())

    # Check that the couplings are anti-symetric
    for mtx in tensor_couplings[:]:
        assertion(np.allclose(mtx, -mtx.T))

    # Check that there are not NaN
    assertion.truth(not np.all(np.isnan(tensor_couplings)))


def check_hamiltonians(hamiltonians: Sequence[str]) -> None:
    """Check that the hamiltonians were written correctly."""
    energies = np.stack([np.diag(np.loadtxt(ts[1])) for ts in hamiltonians])
    couplings = np.stack([np.loadtxt(ts[0]) for ts in hamiltonians])

    # check that energies and couplings are finite values
    assertion.truth(np.isfinite(energies).all())
    assertion.truth(np.isfinite(couplings).all())

    # Check that the couplings diagonal is zero
    assertion.truth(abs(np.einsum('jii->', couplings)) < 1e-16)
