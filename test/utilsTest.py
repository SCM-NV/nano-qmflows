
from functools import wraps
from nac.basisSet.basisNormalization import createNormalizedCGFs
from nac.common import InputKey
from qmworks.parsers.xyzParser import AtomXYZ

import numpy as np
import os

# ========================<>=======================
angs2au = 1 / 0.529177249


def try_to_remove(path_file='test/test_files/test.hdf5'):
    """
    Decorator to remove the intermediate data created during the testing.
    """
    def remove_test_files(fun):
        @wraps
        def wrapper(*args, **kwargs):
            try:
                result = fun(*args, **kwargs)
            finally:
                if os.path.exist(path_file):
                    os.remove(path_file)
            return result
        return wrapper
    return remove_test_files


def create_dict_CGFs(f5, packageHDF5, pathBasis, basisname, packageName, xyz):
    """
    If the Cp2k Basis are already stored in the hdf5 file continue,
    otherwise read and store them in the hdf5 file
    """
    keyBasis = InputKey("basis", [pathBasis])
    packageHDF5(f5, [keyBasis])   # Store the basis sets
    return  createNormalizedCGFs(f5, basisname, packageName, xyz)


def dump_MOs_coeff(handle_hdf5, packageHDF5, path_MO, pathEs, pathCs, nOrbitals):
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


def change_mol_units(mol, factor=angs2au):
    """change the units of the molecular coordinates"""
    newMol = []
    for atom in mol:
        coord = list(map(lambda x: x * factor, atom.xyz))
        newMol.append(AtomXYZ(atom.symbol, coord))
    return newMol


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


def try_to_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def format_aomix(mtx, dim):
    xs = "[overlap matrix]\n"
    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            if j <= i:
                v = mtx[i - 1, j - 1]
                xs += '({:13}  ,{:13} ) = {:25.15e}\n'.format(i, j, v)
    return xs
