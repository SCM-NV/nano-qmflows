"""Functions use for testing."""

import re
import subprocess
import textwrap
import fnmatch
import os
import shutil
from pathlib import Path

import pytest
from qmflows.packages import Result
from qmflows.test_utils import find_executable
from packaging.version import Version

__all__ = [
    "PATH_TEST",
    "CP2K_VERSION",
    "remove_files",
    "validate_status",
    "requires_cp2k",
]

# Environment data
CP2K_EXEC = "cp2k.ssmp"
ROOT = Path(__file__).parents[1]
PATH_TEST = ROOT / "test" / "test_files"

#: A mark for skipping tests if CP2K is not installed
requires_cp2k = pytest.mark.skipif(
    find_executable(CP2K_EXEC) is None,
    reason="Requires CP2K",
)


def _get_cp2K_version(executable: str) -> Version:
    path = find_executable(executable)
    if path is None:
        return Version("0.0")

    out = subprocess.run(
        f"{path} --version",
        check=True, capture_output=True, text=True, shell=True,
    )

    match = re.search(r"CP2K version\s+(\S+)", out.stdout)
    if match is None:
        raise ValueError(f"Failed to parse the `{path!r} --version` output:\n\n{out.stdout}")
    return Version(match[1])


# Environment data
CP2K_VERSION = _get_cp2K_version(CP2K_EXEC)


def remove_files() -> None:
    """Remove tmp files in cwd."""
    for path in fnmatch.filter(os.listdir('.'), "plams_workdir*"):
        shutil.rmtree(path)
    for ext in ("hdf5", "db", "lock"):
        name = f"cache.{ext}"
        if os.path.exists(name):
            os.remove(name)


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
        with open(i, "r", encoding="utf8") as f:
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
