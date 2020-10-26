"""Functions use for testing."""
import fnmatch
import os
import shutil
from distutils.spawn import find_executable
from os.path import join
from pathlib import Path
from typing import Union

import h5py
import pkg_resources as pkg

__all__ = ["PATH_NANOQM", "PATH_TEST", "copy_basis_and_orbitals", "cp2k_available", "remove_files"]

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
