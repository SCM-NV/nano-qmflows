
__all__ = ['AtomBasisData', 'AtomBasisKey', 'AtomData', 'AtomXYZ', 'CGF',
           'InfoMO', 'InputKey', 'MO',
           'binomial', 'change_mol_units', 'even',
           'fac', 'getmass', 'odd', 'product', 'retrieve_hdf5_data',
           'triang2mtx']

# ================> Python Standard  and third-party <==========
from collections import namedtuple
from functools import reduce
import h5py
import numpy as np
import operator as op

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


def getmass(s):
    d = {'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8,
         'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15,
         's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22,
         'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29,
         'zn': 30, 'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36,
         'rb': 37, 'sr': 38, 'Y': 39, 'zr': 40, 'nb': 41, 'mo': 42,
         'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48,
         'in': 49, 'sn': 50, 'sb': 51, 'te': 52, 'i': 53, 'xe': 54, 'cs': 55}

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


def fromIndex(ixs, shape):
    """
    calculate the equivalent index from a two dimensional array to a flat array
    containing the upper triangular elements of a matrix.
    """
    i, j = ixs
    if j >= i:
        k = sum(m * k for m, k in zip(shape[1:], ixs)) + ixs[-1]
        r = (((i * i + i) // 2) if i else 0)
        return k - r
    else:
        return fromIndex([j, i], shape)


def triang2mtx(arr, dim):
    """
    Transform a symmetric matrix represented as a flatten upper triangular
    matrix to the correspoding 2-dimensional array.
    """
    rss = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            k = fromIndex([i, j], [dim, dim])
            rss[i, j] = arr[k]
    return rss


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
