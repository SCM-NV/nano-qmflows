"""Functions use for testing."""
import fnmatch
import os
import shutil
from distutils.spawn import find_executable
from os.path import join
from pathlib import Path

import h5py
import pkg_resources as pkg
from qmflows.type_hints import PathLike

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


def cp2k_available() -> None:
    """Check if cp2k is installed."""
    path = find_executable("cp2k.popt")

    return path is not None


def copy_basis_and_orbitals(source: PathLike, dest, project_name: PathLike) -> None:
    """Copy the Orbitals and the basis set from one the HDF5 to another."""
    keys = [project_name, 'cp2k']
    excluded = ['multipole', 'coupling', 'dipole_matrices',
                'overlaps', 'swaps', 'omega_xia']
    with h5py.File(source, 'r') as f5, h5py.File(dest, 'w') as g5:
        for k in keys:
            if k not in g5:
                g5.create_group(k)
            for l in f5[k].keys():
                if not any(x in l for x in excluded):
                    path = join(k, l)
                    f5.copy(path, g5[k])
