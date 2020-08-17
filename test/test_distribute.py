"""Test the distribution script."""
from qmflows.type_hints import PathLike
from pathlib import Path
from subprocess import (PIPE, Popen)
import fnmatch
import shutil
import os


def test_distribute(tmp_path: PathLike) -> None:
    """Check that the scripts to compute a trajectory are generated correctly."""
    cmd1 = "distribute_jobs.py -i test/test_files/input_test_distribute_derivative_couplings.yml"
    cmd2 = "distribute_jobs.py -i test/test_files/input_test_distribute_absorption_spectrum.yml"
    for cmd in [cmd1, cmd2]:
        call_distribute(tmp_path, cmd)


def call_distribute(tmp_path: PathLike, cmd: str) -> None:
    """Execute the distribute script and check that if finish succesfully."""
    try:
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        _, err = p.communicate()
        if err:
            raise RuntimeError(err.decode())
        check_scripts()
    finally:
        remove_chunk_folder()


def check_scripts() -> None:
    """Check that the distribution scripts were created correctly."""
    paths = fnmatch.filter(os.listdir('.'), "chunk*")

    # Check that the files are created correctly
    files = ["launch.sh", "chunk_xyz*", "input.yml"]
    for p in paths:
        p = Path(p)
        for f in files:
            try:
                next(p.glob(f))
            except StopIteration:
                msg = f"There is not file: {f}"
                print(msg)
                raise RuntimeError(msg)


def remove_chunk_folder() -> None:
    """Remove resulting scripts."""
    for path in fnmatch.filter(os.listdir('.'), "chunk*"):
        shutil.rmtree(path)
