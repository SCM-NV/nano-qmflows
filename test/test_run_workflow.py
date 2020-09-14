"""Test the CLI to run a workflow."""

import os
import re
import shutil
from pathlib import Path
from subprocess import PIPE, Popen

import yaml

from .utilsTest import PATH_TEST


def copy_files(tmp_path: Path) -> None:
    """Copy the input and the hdf5 file to the tmp folder."""
    # Read original input
    path = PATH_TEST / "input_test_single_points.yml"
    with open(path, 'r') as f:
        single_point = yaml.load(f, yaml.FullLoader)

    tmp_hdf5 = (tmp_path / "ethylene.hdf5").as_posix()

    # copy hdf5 to temporal folder
    shutil.copy(single_point["path_hdf5"], tmp_path)

    # assign temporal HDF5
    single_point["path_hdf5"] = tmp_hdf5

    # Copy input to temporal file
    with open(tmp_path / "single_point.yml", 'w') as f:
        yaml.dump(single_point, f)


def test_run_workflow(tmp_path):
    """Test the run_workflow functionality."""
    copy_files(Path(tmp_path))
    cmd = f"run_workflow.py -i {tmp_path}/single_point.yml"
    try:
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        out, err = p.communicate()
        error = re.search("error", err.decode(), re.IGNORECASE)
        if error is not None:
            print("output: ", out)
            print("err: ", err)
            raise RuntimeError(err.decode())
    finally:
        files = map(lambda p: Path(p), ("cache.db", "ethylene.log"))
        for f in files:
            if f.exists():
                os.remove(f.absolute().as_posix())
