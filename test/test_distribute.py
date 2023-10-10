"""Test the distribution script."""

import fnmatch
import shutil
import os
import pytest
from pathlib import Path
from typing import Literal

from nanoqm.workflows.distribute_jobs import distribute_jobs
from nanoqm.workflows.input_validation import process_input

_WorkflowKind = Literal["derivative_couplings", "absorption_spectrum"]

JOBS = {
    "derivative_couplings": "test/test_files/input_test_distribute_derivative_couplings.yml",
    "absorption_spectrum": "test/test_files/input_test_distribute_absorption_spectrum.yml",
}


@pytest.mark.parametrize("workflow,file", JOBS.items(), ids=JOBS)
def test_distribute(workflow: _WorkflowKind, file: str) -> None:
    """Execute the distribute script and check that if finish succesfully."""
    try:
        distribute_jobs(file)
        check_scripts(workflow)
    finally:
        remove_chunk_folder()


def check_scripts(workflow: _WorkflowKind) -> None:
    """Check that the distribution scripts were created correctly."""
    paths = fnmatch.filter(os.listdir('.'), "chunk*")
    cwd_old = os.getcwd()

    # Check that the files are created correctly
    files = ["launch.sh", "chunk_xyz*", "input.yml"]
    for _p in paths:
        p = Path(_p)
        for f in files:
            try:
                next(p.glob(f))
            except StopIteration:
                msg = f"There is no such file: {f!r}"
                raise RuntimeError(msg) from None
            if f == "input.yml":
                os.chdir(p)
                try:
                    process_input(f, workflow)
                finally:
                    os.chdir(cwd_old)


def remove_chunk_folder() -> None:
    """Remove resulting scripts."""
    for path in fnmatch.filter(os.listdir('.'), "chunk*"):
        shutil.rmtree(path, ignore_errors=True)
