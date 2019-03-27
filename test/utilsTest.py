from distutils.spawn import find_executable
from os.path import join
import h5py

import fnmatch
import shutil
import os


def remove_files():
    """ Remove tmp files in cwd """
    for path in fnmatch.filter(os.listdir('.'), "plams_workdir*"):
        shutil.rmtree(path)
    if os.path.exists("cache.db"):
        os.remove("cache.db")


def cp2k_available():
    """
    Check if cp2k is installed
    """
    path = find_executable("cp2k.popt")

    return path is not None


def copy_basis_and_orbitals(source, dest, project_name):
    """
    Copy the Orbitals and the basis set from one the HDF5 to another
    """
    keys = [project_name, 'cp2k']
    excluded = ['multipole', 'coupling', 'dipole_matrices', 'overlaps', 'swaps']
    with h5py.File(source, 'r') as f5, h5py.File(dest, 'w') as g5:
        for k in keys:
            if k not in g5:
                g5.create_group(k)
            for l in f5[k].keys():
                if not any(x in l for x in excluded):
                    path = join(k, l)
                    f5.copy(path, g5[k])
