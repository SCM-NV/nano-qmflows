"""Test the derivative coupling calculation."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence, Generator, Generator
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import h5py
import numpy as np
import pytest
from assertionlib import assertion
from nanoqm.common import DictConfig, is_data_in_hdf5, retrieve_hdf5_data
from nanoqm.workflows.input_validation import process_input
from nanoqm.workflows.workflow_coupling import workflow_derivative_couplings

from .utilsTest import PATH_TEST, remove_files

if TYPE_CHECKING:
    import _pytest


class CouplingsOutput(NamedTuple):
    name: str
    config: DictConfig
    tmp_hdf5: Path
    orbitals_type: str
    hamiltonians: tuple[list[str], ...]


class TestCoupling:
    PARAMS = {
        "couplings": ("input_fast_test_derivative_couplings.yml", ""),
        "couplings_alphas": ("input_couplings_alphas.yml", "alphas"),
        "couplings_both": ("input_couplings_both.yml", "both"),
    }

    @pytest.fixture(scope="class", params=PARAMS.items(), ids=PARAMS, name="output")
    def get_couplings(
        self, request: _pytest.fixtures.SubRequest
    ) -> Generator[CouplingsOutput, None, None]:
        name, (input_file, orbitals_type) = request.param

        with tempfile.TemporaryDirectory() as _tmp_path:
            tmp_path = Path(_tmp_path)

            path_input = PATH_TEST / input_file
            config = process_input(path_input, 'derivative_couplings')
            config["scratch_path"] = tmp_path
            tmp_hdf5 = os.path.join(tmp_path, 'fast_couplings.hdf5')
            shutil.copy(config.path_hdf5, tmp_hdf5)
            config['path_hdf5'] = tmp_hdf5
            config['write_overlaps'] = True

            # Run the calculation again to test that the data is read from the hdf5
            _ = workflow_derivative_couplings(config)
            output = workflow_derivative_couplings(config)
            if orbitals_type != "both":
                hamiltonians = (output[0],)
            else:
                hamiltonians = (output[0][0], output[1][0])

            yield CouplingsOutput(name, config, tmp_hdf5, orbitals_type, hamiltonians)

        # Teardown
        remove_files()

    def test_couplings(self, output: CouplingsOutput) -> None:
        if output.orbitals_type != "both":
            self._test_couplings(output, output.orbitals_type)
        else:
            self._test_couplings(output, "alphas")
            self._test_couplings(output, "betas")

    def _test_couplings(self, output: CouplingsOutput, orbitals_type: str) -> None:
        """Check that the couplings have meaningful values."""
        def create_paths(keyword: str) -> list:
            config_range = range(len(output.config.geometries) - 1)
            return [os.path.join(orbitals_type, f'{keyword}_{x}') for x in config_range]

        _overlaps_path_product = product(create_paths('overlaps'), ['mtx_sji_t0', 'mtx_sji_t0_corrected'])
        overlaps_path = [f"{i}/{j}" for i, j in _overlaps_path_product]
        couplings_path = create_paths('coupling')

        # Check that couplings and overlaps exists
        assertion.assert_(is_data_in_hdf5, output.tmp_hdf5, overlaps_path, message="overlaps dataset")
        assertion.assert_(is_data_in_hdf5, output.tmp_hdf5, couplings_path, message="couplings dataset")
        overlaps = np.array(retrieve_hdf5_data(output.tmp_hdf5, overlaps_path))
        couplings = np.array(retrieve_hdf5_data(output.tmp_hdf5, couplings_path))

        # All the elements are different of inifinity or nan
        assertion.assert_(np.isreal, couplings, post_process=np.all)

        # Check that the couplings are anti-symetric
        for mtx in couplings:
            np.testing.assert_allclose(mtx, -mtx.T, rtol=0, atol=1e-8)

        # Compare with reference data
        if orbitals_type in {"alphas", "betas"}:
            name = f"{output.name}-{orbitals_type}"
        else:
            name = output.name
        with h5py.File(PATH_TEST / "test_files.hdf5", "r") as f:
            ref_couplings = f[f"test_coupling/TestCoupling/{name}/couplings"][...]
            ref_overlaps = f[f"test_coupling/TestCoupling/{name}/overlaps"][...]
        np.testing.assert_allclose(couplings, ref_couplings, rtol=0, atol=1e-06)
        np.testing.assert_allclose(overlaps, ref_overlaps, rtol=0, atol=1e-06)

    def test_hamiltonians(self, output: CouplingsOutput) -> None:
        if len(output.hamiltonians) == 1:
            names = [output.name]
            suffices = [""]
        else:
            names = [f"{output.name}-{i}" for i in ["alphas", "betas"]]
            suffices = [" (alphas)", " (betas)"]
        for ham, name, suffix in zip(output.hamiltonians, names, suffices):
            self._test_hamiltonians(ham, name, suffix)

    def _test_hamiltonians(self, hamiltonians: Sequence[str], name: str, suffix: str) -> None:
        """Check that the hamiltonians were written correctly."""
        energies = np.stack([np.diag(np.loadtxt(ts[1])) for ts in hamiltonians])
        couplings = np.stack([np.loadtxt(ts[0]) for ts in hamiltonians])

        # check that energies and couplings are finite values
        assertion.assert_(np.isfinite, energies, post_process=np.all, message=f"energies{suffix}")
        assertion.assert_(np.isfinite, couplings, post_process=np.all, message=f"couplings{suffix}")

        # Check that the couplings diagonal is zero
        trace = np.trace(couplings, axis1=1, axis2=2)
        np.testing.assert_allclose(trace, 0.0, err_msg=f"couplings trace{suffix}")

        # Compare with reference data
        with h5py.File(PATH_TEST / "test_files.hdf5", "r") as f:
            ref_energies = f[f"test_coupling/TestCoupling/{name}/txt_energies"][...]
            ref_couplings = f[f"test_coupling/TestCoupling/{name}/txt_couplings"][...]
        np.testing.assert_allclose(energies, ref_energies, rtol=0, atol=1e-06)
        np.testing.assert_allclose(couplings, ref_couplings, rtol=0, atol=1e-06)
