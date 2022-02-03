"""Functions use for testing."""

import textwrap
import fnmatch
import os
import shutil
from distutils.spawn import find_executable
from os.path import join
from pathlib import Path
from typing import Union

import h5py
import pkg_resources as pkg
from qmflows.packages.packages import Result

__all__ = [
    "PATH_NANOQM",
    "PATH_TEST",
    "copy_basis_and_orbitals",
    "cp2k_available",
    "remove_files",
    "validate_status",
]

# Environment data
PATH_NANOQM = Path(pkg.resource_filename('nanoqm', ''))
ROOT = PATH_NANOQM.parent
PATH_TEST = ROOT / "test" / "test_files"


def remove_files() -> None:
    """Remove tmp files in cwd."""
    for path in fnmatch.filter(os.listdir('.'), "plams_workdir*"):
        shutil.rmtree(path)
    for ext in ("hdf5", "db", "lock"):
        name = f"cache.{ext}"
        if os.path.exists(name):
            os.remove(name)


def cp2k_available(executable: str = "cp2k.popt") -> bool:
    """Check if cp2k is installed."""
    path = find_executable(executable)

    return path is not None


def _read_result_file(result: Result, extension: str, max_line: int = 100) -> "None | str":
    """Find and read the first file in ``result`` with the provided file extension.

    Returns ``None`` if no such file can be found.
    """
    root = result.archive["plams_dir"]
    if root is None:
        return None

    iterator = (os.path.join(root, i) for i in os.listdir(root)
                if os.path.splitext(i)[1] == extension)
    for i in iterator:
        with open(i, "r") as f:
            ret_list = f.readlines()
            ret = "..." if len(ret_list) > max_line else ""
            ret += "".join(ret_list[-max_line:])
            return textwrap.indent(ret, 4 * " ")
    else:
        return None


def validate_status(result: Result, *, print_out: bool = True, print_err: bool = True) -> None:
    """Validate the status of the ``qmflows.Result`` object is set to ``"successful"``.

    Parameters
    ----------
    result : qmflows.Result
        The to-be validated ``Result`` object.
    print_out : bool
        Whether to included the content of the ``Result`` objects' .out file in the exception.
    print_err : bool
        Whether to included the content of the ``Result`` objects' .err file in the exception.

    Raises
    ------
    AssertionError
        Raised when :code:`result.status != "successful"`.

    """
    # TODO: Import `validate_status` from qmflows once 0.11.2 has been release
    if result.status == "successful":
        return None

    indent = 4 * " "
    msg = f"Unexpected {result.job_name} status: {result.status!r}"

    if print_out:
        out = _read_result_file(result, ".out")
        if out is not None:
            msg += f"\n\nout_file:\n{out}"
    if print_err:
        err = _read_result_file(result, ".err")
        if err is not None:
            msg += f"\n\nerr_file:\n{err}"
    raise AssertionError(msg)
