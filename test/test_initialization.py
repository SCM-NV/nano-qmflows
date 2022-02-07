"""Test that the path are created propery."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import pytest
import yaml
import pkg_resources as pkg
from nanoutils import RecursiveKeysView
from assertionlib import assertion
from qmflows.parsers.cp2KParser import readCp2KBasis

from nanoqm.common import UniqueSafeLoader, DictConfig
from nanoqm.workflows.initialization import initialize, save_basis_to_hdf5
from nanoqm.workflows.input_validation import process_input
from .utilsTest import PATH_TEST

if TYPE_CHECKING:
    import _pytest


def test_run_workflow(tmp_path: Path) -> None:
    """Check that all the paths are initialize."""
    create_config(tmp_path, True)
    create_config(tmp_path, False)


def create_config(tmp_path: Path, scrath_is_None: bool) -> None:
    path = PATH_TEST / "input_fast_test_derivative_couplings.yml"
    with open(path, 'r') as f:
        inp = yaml.load(f, UniqueSafeLoader)

    # change scratch
    if scrath_is_None:
        inp["scratch_path"] = None
        del inp["path_hdf5"]
    else:
        inp["scratch_path"] = (tmp_path / "level0" / "level1").as_posix()
        # change HDF5 path
        inp["path_hdf5"] = (Path(inp["scratch_path"]) / "test_init.hdf5").as_posix()

    path_inp = tmp_path / "test_init.yml"
    with open(path_inp, 'w') as f:
        yaml.dump(inp, f)

    new_inp = process_input(path_inp, 'derivative_couplings')

    config = initialize(new_inp)

    assert Path(config.path_hdf5).exists()


class TestSaveBasisToHDF5:
    """Test ``save_basis_to_hdf5``."""

    PARAM = {
        "None": None,
        "MOLOPT": ["BASIS_MOLOPT"],
        "MOLOPT_UZH": ["BASIS_MOLOPT", "BASIS_MOLOPT_UZH"],
        "ADMM": ["BASIS_MOLOPT", "BASIS_ADMM", "BASIS_ADMM_MOLOPT"],
    }

    @pytest.fixture(scope="function", autouse=True, params=PARAM.items(), ids=PARAM, name="input")
    def get_input(
        self,
        request: _pytest.fixtures.SubRequest,
        tmp_path: Path,
    ) -> tuple[DictConfig, set[str]]:
        name, basis_file_name = request.param
        if name == "ADMM":  # TODO
            pytest.xfail("Basis sets consisting of multiple subsets aren't supported yet")

        # COnstruct the settings
        hdf5_file = tmp_path / f"{name}.hdf5"
        config = DictConfig(
            path_hdf5=hdf5_file,
            cp2k_general_settings=DictConfig(
                path_basis=PATH_TEST,
                basis_file_name=basis_file_name,
            ),
        )

        # Ensure that a fresh .hdf5 is created
        if os.path.isfile(hdf5_file):
            os.remove(hdf5_file)
        with h5py.File(hdf5_file, "w-"):
            pass

        # Construct a set with all keys that are supposed to be in the .hdf5 file
        with open(PATH_TEST / "test_initialization.yaml", "r") as f:
            keys = set(yaml.load(f, Loader=yaml.SafeLoader)[name])
        return config, keys

    def test_pass(self, input: tuple[DictConfig, set[str]]) -> None:
        config, ref = input

        save_basis_to_hdf5(config)

        with h5py.File(config.path_hdf5, "r") as f:
            assertion.eq(RecursiveKeysView(f), ref)
