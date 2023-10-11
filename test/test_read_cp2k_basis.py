"""Read basis in CP2K format."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import yaml
import h5py
from nanoutils import RecursiveKeysView, RecursiveItemsView
from packaging.version import Version
from assertionlib import assertion

import nanoqm
from nanoqm.workflows.initialization import store_cp2k_basis
from .utilsTest import PATH_TEST


class TestRedCP2KBasis:
    def test_pass(self, tmp_path: Path) -> None:
        """Read Basis set in CP2K format."""
        tmp_hdf5 = Path(tmp_path) / 'cp2k_basis.hdf5'
        with h5py.File(tmp_hdf5, "x"):
            pass

        path_basis = os.path.join(
            os.path.dirname(nanoqm.__file__),
            'basis',
            'BASIS_MOLOPT',
        )

        store_cp2k_basis(tmp_hdf5, path_basis)

        with h5py.File(tmp_hdf5, 'r') as f5:
            dset = f5["cp2k/basis/c/DZVP-MOLOPT-GTH/0/coefficients"]

            # Check that the format is store
            ref = [2, 0, 2, 7, 2, 2, 1]
            np.testing.assert_array_equal(dset.attrs['basisFormat'], ref)

            # Check Shape of the coefficients
            assertion.eq(dset.shape, (5, 7))

    def test_legacy(self, tmp_path: Path) -> None:
        hdf5_file = tmp_path / "legacy.hdf5"
        shutil.copy2(PATH_TEST / "legacy.hdf5", hdf5_file)

        store_cp2k_basis(hdf5_file, PATH_TEST / "BASIS_MOLOPT")
        with open(PATH_TEST / "test_initialization.yaml", "r", encoding="utf8") as f1:
            ref = set(yaml.load(f1, Loader=yaml.SafeLoader)["MOLOPT"])

        with h5py.File(hdf5_file, "r") as f2:
            assertion.eq(RecursiveKeysView(f2), ref)
            assertion.assert_(Version, f2.attrs.get("__version__"))

            for name, dset in RecursiveItemsView(f2):
                if not name.endswith("coefficients"):
                    continue
                basis_fmt = dset.attrs.get("basisFormat")
                assertion.isinstance(basis_fmt, np.ndarray, message=name)
                assertion.eq(basis_fmt.ndim, 1, message=name)
                assertion.eq(basis_fmt.dtype, np.int64, message=name)
