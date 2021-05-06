"""Test that the path are created propery."""

from pathlib import Path
from typing import Optional
import yaml

from nanoqm.common import UniqueSafeLoader
from nanoqm.workflows.initialization import initialize
from nanoqm.workflows.input_validation import process_input
from qmflows.type_hints import PathLike

from .utilsTest import PATH_TEST


def test_run_workflow(tmp_path: PathLike) -> None:
    """Check that all the paths are initialize."""
    create_config(tmp_path, True)
    create_config(tmp_path, False)


def create_config(tmp_path: str, scrath_is_None: bool) -> str:
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
