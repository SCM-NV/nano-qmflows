"""Test he absorption spectrum workflows."""
import shutil
from os.path import join
from pathlib import Path

import h5py
import pytest
import qmflows
import numpy as np
from packaging.version import Version
from nanoqm.common import retrieve_hdf5_data
from nanoqm.workflows import workflow_stddft
from nanoqm.workflows.input_validation import process_input

from .utilsTest import PATH_TEST, remove_files


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
        approx: str,
    ) -> None:
        """Compute the oscillator strenght and check the results."""
        if project == "He" and (Version(qmflows.__version__) < Version("0.11.3")):
            pytest.skip("Requires QMFlows >= 0.11.3")

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

    def calculate_oscillators(self, path: Path, scratch: Path, approx: str, inp: str) -> None:
        """Compute a couple of couplings with the Levine algorithm using precalculated MOs."""
        input_file = PATH_TEST / inp
        config = process_input(input_file, 'absorption_spectrum')
        config['path_hdf5'] = path.absolute().as_posix()
        config['scratch_path'] = scratch
        config['workdir'] = scratch
        config['tddft'] = approx
        config['path_traj_xyz'] = Path(config.path_traj_xyz).absolute().as_posix()
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
        np.testing.assert_allclose(dipole_matrices, ref)
