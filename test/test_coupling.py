"""Test the derivative coupling calculation."""
import os
import shutil
from typing import Sequence

import numpy as np

from nanoqm.common import DictConfig, is_data_in_hdf5, retrieve_hdf5_data
from nanoqm.workflows.input_validation import process_input
from nanoqm.workflows.workflow_coupling import workflow_derivative_couplings

from .utilsTest import PATH_TEST, remove_files


def test_fast_couplings(tmp_path):
    """Check the derivative couplings workflow"""
    run_derivative_coupling(tmp_path, 'input_fast_test_derivative_couplings.yml')


def test_unrestricted_alphas(tmp_path):
    """Test the derivative coupling for the alphas spin orbitals."""
    run_derivative_coupling(tmp_path, 'input_couplings_alphas.yml')


def test_unrestricted_betas(tmp_path):
    """Test the derivative coupling for the alphas spin orbitals."""
    run_derivative_coupling(tmp_path, 'input_couplings_betas.yml')


def run_derivative_coupling(tmp_path: str, input_file: str) -> None:
    """Check that the couplings run."""
    path_input = PATH_TEST / input_file
    config = process_input(path_input, 'derivative_couplings')
    config["scratch_path"] = tmp_path
    tmp_hdf5 = os.path.join(tmp_path, 'fast_couplings.hdf5')
    shutil.copy(config.path_hdf5, tmp_hdf5)
    config['path_hdf5'] = tmp_hdf5
    try:
        hamiltonians, _ = workflow_derivative_couplings(config)
        check_couplings(config, tmp_hdf5)
        check_hamiltonians(hamiltonians)
    finally:
        remove_files()


def check_couplings(config: DictConfig, tmp_hdf5: str) -> None:
    """Check that the couplings have meaningful values."""
    def create_paths(keyword: str) -> list:
        return [os.path.join(config.project_name, config.orbitals_type, f'{keyword}_{x}')
                for x in range(len(config.geometries) - 1)]
    overlaps = create_paths('overlaps')
    couplings = create_paths('coupling')

    # Check that couplings and overlaps exists
    assert is_data_in_hdf5(tmp_hdf5, overlaps)
    assert is_data_in_hdf5(tmp_hdf5, couplings)

    # All the elements are different of inifinity or nan
    tensor_couplings = np.stack(retrieve_hdf5_data(tmp_hdf5, couplings))
    assert np.isfinite(tensor_couplings).all()

    # Check that the couplings are anti-symetric
    for mtx in tensor_couplings[:]:
        assert np.allclose(mtx, -mtx.T)

    # Check that there are not NaN
    assert (not np.all(np.isnan(tensor_couplings)))


def check_hamiltonians(hamiltonians: Sequence[str]) -> None:
    """Check that the hamiltonians were written correctly."""
    energies = np.stack([np.diag(np.loadtxt(ts[1])) for ts in hamiltonians])
    couplings = np.stack([np.loadtxt(ts[0]) for ts in hamiltonians])

    # check that energies and couplings are finite values
    assert np.isfinite(energies).all()
    assert np.isfinite(couplings).all()

    # Check that the couplings diagonal is zero
    assert abs(np.einsum('jii->', couplings)) < 1e-16
