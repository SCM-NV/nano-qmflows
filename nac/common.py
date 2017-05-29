
__all__ = ['Array', 'AtomBasisData', 'AtomBasisKey', 'AtomData', 'AtomXYZ',
           'CGF', 'InfoMO', 'InputKey', 'Matrix', 'MO', 'Tensor3D', 'Vector',
           'binomial', 'change_mol_units', 'even', 'fac', 'getmass', 'h2ev',
           'odd', 'product', 'retrieve_hdf5_data',
           'search_data_in_hdf5', 'store_arrays_in_hdf5', 'triang2mtx']

# ================> Python Standard  and third-party <==========
from collections import namedtuple
from functools import reduce
import h5py
import numpy as np
import operator as op
import os

# ======================================================================
# Named Tuples
AtomData = namedtuple("AtomData", ("label", "coordinates", "cgfs"))
AtomBasisKey = namedtuple("AtomBasisKey", ("atom", "basis", "basisFormat"))
AtomBasisData = namedtuple("AtomBasisData", ("exponents", "coefficients"))
AtomXYZ = namedtuple("AtomXYZ", ("symbol", "xyz"))
CGF = namedtuple("CGF", ("primitives", "orbType"))
InfoMO = namedtuple("InfoMO", ("eigenVals", "coeffs"))
InputKey = namedtuple("InpuKey", ("name", "args"))
MO = namedtuple("MO", ("coordinates", "cgfs", "coefficients"))

# ================> Constants <================
angs2au = 1 / 0.529177249  # Angstrom to a.u
femtosec2au = 1 / 2.41888432e-2  # from femtoseconds to au
r2meV = 13605.698  # conversion from rydberg to meV
fs_to_cm = 33356.40952  # conversion from fs to cm-1
fs_to_nm = 299.79246  # conversion from fs to nm
hbar = 0.6582119  # planck constant in eV * fs
h2ev = 27.2114  # hartrees to electronvolts

# Numpy type hints
Array = np.ndarray  # Generic Array
Vector = np.ndarray
Matrix = np.ndarray
Tensor3D = np.ndarray


def getmass(s: str):
    d = {'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8,
         'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15,
         's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22,
         'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29,
         'zn': 30, 'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36,
         'rb': 37, 'sr': 38, 'y': 39, 'zr': 40, 'nb': 41, 'mo': 42, 'tc': 43,
         'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48, 'in': 49, 'sn': 50,
         'sb': 51, 'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56, 'la': 57,
         'ce': 58, 'pr': 59, 'nd': 60, 'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64,
         'tb': 65, 'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70, 'lu': 71,
         'hf': 72, 'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78,
         'au': 79, 'hg': 80, 'tl': 81, 'pb': 82, 'bi': 83, 'po': 84, 'at': 85,
         'rn': 86, 'fr': 87, 'ra': 88, 'ac': 89, 'th': 90, 'pa': 91, 'u': 92,
         'np': 93, 'pu': 94, 'am': 95, 'cm': 96, 'bk': 97, 'cf': 98, 'es': 99,
         'fm': 100, 'md': 101, 'no': 102, 'lr': 103, 'rf': 104, 'db': 105}
    return d[s]


def retrieve_hdf5_data(path_hdf5, paths_to_prop):
    """
    Read Numerical properties from ``paths_hdf5``.

    :params path_hdf5: Path to the hdf5 file
    :type path_hdf5: string
    :returns: numerical array

    """
    try:
        with h5py.File(path_hdf5, 'r') as f5:
            if isinstance(paths_to_prop, list):
                return [f5[path].value for path in paths_to_prop]
            else:
                return f5[paths_to_prop].value
    except KeyError:
        msg = "There is not {} stored in the HDF5\n".format(paths_to_prop)
        raise KeyError(msg)
    except FileNotFoundError:
        msg = "there is not HDF5 file containing the numerical results"
        raise RuntimeError(msg)


def search_data_in_hdf5(path_hdf5, xs):
    """
    Search if the node exists in the HDF5 file.
    """
    if os.path.exists(path_hdf5):
        with h5py.File(path_hdf5, 'r') as f5:
            if isinstance(xs, list):
                return all(path in f5 for path in xs)
            else:
                return xs in f5
    else:
        return False


def store_arrays_in_hdf5(path_hdf5: str, paths, tensor: Array,
                         dtype=np.float32)-> None:
    """
    Store the corrected overlaps in the HDF5 file
    """
    with h5py.File(path_hdf5, 'r+') as f5:
        if isinstance(paths, list):
            for k, path in enumerate(paths):
                data = tensor[k]
                f5.require_dataset(path, shape=np.shape(data),
                                   data=data, dtype=dtype)
        else:
            f5.require_dataset(paths, shape=np.shape(tensor),
                               data=tensor, dtype=dtype)


def triang2mtx(xs: Vector, dim: int) -> Matrix:
    """
    Transform a symmetric matrix represented as a flatten upper triangular
    matrix to the correspoding 2-dimensional array.
    """
    # New array
    mtx = np.zeros((dim, dim))
    # indexes of the upper triangular
    inds = np.triu_indices_from(mtx)
    # Fill the upper triangular of the new array
    mtx[inds] = xs
    # Fill the lower triangular
    mtx[(inds[1], inds[0])] = xs

    return mtx


def change_mol_units(mol, factor=angs2au):
    """
    change the units of the molecular coordinates
    :returns: New XYZ namedtuple
    """
    newMol = []
    for atom in mol:
        coord = list(map(lambda x: x * factor, atom.xyz))
        newMol.append(AtomXYZ(atom.symbol, coord))
    return newMol

# Utilities


def product(xs):
    return reduce(op.mul, xs)


def odd(x):
    return x % 2 != 0


def even(x):
    return not(odd(x))


def fac(x):
    if x == 0:
        return 1
    else:
        return float(product(range(1, x + 1)))


def binomial(n, k):
    if k == n:
        return 1.0
    elif k >= 0 and k < n:
        return fac(n) / (fac(k) * fac(n - k))
    else:
        return 0.0
