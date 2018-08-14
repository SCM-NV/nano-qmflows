
from functools import wraps
from nac.basisSet.basisNormalization import create_normalized_CGFs
from nac.common import InputKey
from os.path import join

import h5py
import numpy as np
import os
import shutil

# ========================<>=======================
angs2au = 1 / 0.529177249


def remove_file_directory(paths):
    """
    Remove both a file or a directory
    """
    for p in paths:
        if os.path.exists(p):
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)


def try_to_remove(path_files=None):
    """
    Decorator to remove the intermediate data created during the testing.
    """
    if path_files is None:
        path_files = ['test/test_files/test.hdf5']

    def remove_test_files(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            try:
                return fun(*args, **kwargs)
            finally:
                remove_file_directory(path_files)
        return wrapper
    return remove_test_files


def create_dict_CGFs(f5, packageHDF5, pathBasis, basisname, packageName, xyz):
    """
    If the Cp2k Basis are already stored in the hdf5 file continue,
    otherwise read and store them in the hdf5 file
    """
    keyBasis = InputKey("basis", [pathBasis])
    packageHDF5(f5, [keyBasis])   # Store the basis sets
    return create_normalized_CGFs(f5, basisname, packageName, xyz)


def dump_MOs_coeff(handle_hdf5, packageHDF5, path_MO, pathEs, pathCs,
                   nOrbitals):
    """
    MO coefficients are stored in row-major order, they must be transposed
    to get the standard MO matrix.
    :param files: Files to calculate the MO coefficients
    :type  files: Namedtuple (fileXYZ,fileInput,fileOutput)
    :param job: Output File
    :type  job: String
    """
    key = InputKey('orbitals', [path_MO, nOrbitals, pathEs, pathCs])

    packageHDF5(handle_hdf5, [key])

    return pathEs, pathCs


def copy_basis_and_orbitals(source, dest, project_name):
    """
    Copy the Orbitals and the basis set from one the HDF5 to another
    """
    keys = [project_name, 'cp2k']
    excluded = ['coupling', 'dipole_matrices', 'overlaps', 'swaps']
    with h5py.File(source, 'r') as f5, h5py.File(dest, 'w') as g5:
        for k in keys:
            if k not in g5:
                g5.create_group(k)
            for l in f5[k].keys():
                if not any(x in l for x in excluded):
                    path = join(k, l)
                    f5.copy(path, g5[k])


def offdiagonalTolerance(arr, tolerance=1.0e-8):
    """Check if the off-diagonal entries are lower than a
    numerical tolerance
    """
    dim, _ = np.shape(arr)
    r = False
    for i in range(dim):
        for j in range(dim):
            if i >= j:
                r = r and True
            else:
                r = abs(arr[i, j]) < tolerance
    return r


def fromIndex(ixs, shape):
    i, j = ixs
    if j >= i:
        k = sum(m * k for m, k in zip(shape[1:], ixs)) + ixs[-1]
        r = (((i * i + i) // 2) if i else 0)
        return k - r
    else:
        return fromIndex([j, i], shape)


def triang2mtx(arr, dim):
    rss = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            k = fromIndex([i, j], [dim, dim])
            rss[i, j] = arr[k]
    return rss


def format_aomix(mtx, dim):
    xs = "[overlap matrix]\n"
    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            if j <= i:
                v = mtx[i - 1, j - 1]
                xs += '({:13}  ,{:13} ) = {:25.15e}\n'.format(i, j, v)
    return xs
