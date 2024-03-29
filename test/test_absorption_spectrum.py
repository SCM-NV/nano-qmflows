"""Test he absorption spectrum workflows."""
import shutil
from os.path import join
from pathlib import Path
from typing import Literal

import h5py
import pytest
import numpy as np
from qmflows.warnings_qmflows import Orbital_Warning
from assertionlib import assertion

from nanoqm.common import retrieve_hdf5_data, DictConfig
from nanoqm.workflows import workflow_stddft
from nanoqm.workflows.input_validation import process_input
from nanoqm.workflows.workflow_stddft_spectrum import validate_active_space
from .utilsTest import PATH_TEST, remove_files, requires_cp2k


@requires_cp2k
class TestComputeOscillators:
    _PARAMS = {
        "MOLOPT": ("Cd", "input_test_absorption_spectrum.yml", ""),
        "ALL_BASIS_SETS": ("He", "input_test_absorption_spectrum_all.yml", ""),
        "unrestricted": ("oxygen", "input_test_absorption_spectrum_unrestricted.yml", "alphas"),
    }
    PARAMS = {k: (k, *v) for k, v in _PARAMS.items()}
    del _PARAMS

    @pytest.mark.parametrize("approx", ["sing_orb", "stda"])
    @pytest.mark.parametrize("name,project,inp,orbital_type", PARAMS.values(), ids=PARAMS.keys())
    def test(
        self,
        tmp_path: Path,
        name: str,
        project: str,
        inp: str,
        orbital_type: str,
        approx: Literal["sing_orb", "stda"],
    ) -> None:
        """Compute the oscillator strenght and check the results."""
        name += f"-{approx}"
        path_original_hdf5 = PATH_TEST / f'{project}.hdf5'
        shutil.copy(path_original_hdf5, tmp_path)
        try:
            # Run the actual test
            path = Path(tmp_path) / f"{project}_{approx}.hdf5"
            shutil.copyfile(path_original_hdf5, path)
            self.calculate_oscillators(path, tmp_path, approx, inp)
            self.check_properties(path, orbital_type, name)

            # Run again the workflow to check that the data is read from the hdf5
            self.calculate_oscillators(path, tmp_path, approx, inp)
            self.check_properties(path, orbital_type, name)
        finally:
            remove_files()

    def calculate_oscillators(
        self, path: Path, scratch: Path, approx: Literal["sing_orb", "stda"], inp: str
    ) -> None:
        """Compute a couple of couplings with the Levine algorithm using precalculated MOs."""
        input_file = PATH_TEST / inp
        config = process_input(input_file, 'absorption_spectrum')
        config.path_hdf5 = path.absolute().as_posix()
        config.scratch_path = scratch
        config.workdir = scratch
        config.tddft = approx
        config.path_traj_xyz = Path(config.path_traj_xyz).absolute().as_posix()
        workflow_stddft(config)

    def check_properties(self, path: Path, orbitals_type: str, name: str) -> None:
        """Check that the tensor stored in the HDF5 are correct."""
        path_dipole = join(orbitals_type, 'dipole', 'point_0')
        dipole_matrices = retrieve_hdf5_data(path, path_dipole)

        # The diagonals of each component of the matrix must be zero
        # for a single atom
        trace = dipole_matrices.trace(axis1=1, axis2=2)
        np.testing.assert_allclose(trace[1:], 0.0)

        # Compare with reference data
        with h5py.File(PATH_TEST / "test_files.hdf5", "r") as f:
            ref = f[f"test_absorption_spectrum/TestComputeOscillators/{name}/dipole"][...]
        np.testing.assert_allclose(dipole_matrices, ref, rtol=0, atol=1e-08)


def test_active_space_readjustment() -> None:
    config = DictConfig(
        active_space=(6, 8),
        multiplicity=2,
        orbitals_type="betas",
    )
    with pytest.warns(Orbital_Warning):
        out = validate_active_space(config, 4, 6)
    assertion.eq(out, (4, 6))
