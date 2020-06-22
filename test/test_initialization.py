"""Test that the path are created propery."""

from pathlib import Path

import yaml

from nanoqm.workflows.initialization import initialize
from nanoqm.workflows.input_validation import process_input

from .utilsTest import PATH_TEST


def test_run_workflow(tmp_path):
    """Check that all the paths are initialize."""
    path = PATH_TEST / "input_fast_test_derivative_couplings.yml"
    with open(path, 'r') as f:
        inp = yaml.load(f, yaml.FullLoader)

    # change scratch
    inp["scratch_path"] = (tmp_path / "level0" / "level1").as_posix()

    # change HDF5 path
    inp["path_hdf5"] = (Path(inp["scratch_path"]) / "test_init.hdf5").as_posix()

    path_inp = tmp_path / "test_init.yml"
    with open(path_inp, 'w') as f:
        yaml.dump(inp, f)

    new_inp = process_input(path_inp, 'derivative_couplings')

    config = initialize(new_inp)

    assert Path(config.path_hdf5).exists()
