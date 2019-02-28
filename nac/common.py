
__all__ = ['Array', 'AtomBasisData', 'AtomBasisKey', 'AtomData', 'AtomXYZ',
           'CGF', 'DictConfig', 'InfoMO', 'InputKey', 'Matrix', 'MO', 'Tensor3D', 'Vector',
           'binomial', 'change_mol_units', 'even', 'fac', 'getmass', 'h2ev',
           'hardness', 'odd', 'product', 'retrieve_hdf5_data',
           'search_data_in_hdf5', 'store_arrays_in_hdf5', 'triang2mtx']


from collections import namedtuple
from functools import reduce
from scipy.constants import physical_constants
from scm.plams import (Atom, Molecule)
from typing import (Dict, List, Tuple)

import h5py
import numpy as np
import operator as op
import os


class DictConfig(dict):

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __deepcopy__(self, _):
        return DictConfig(self.copy())


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
# Angstrom to a.u
angs2au = 1e-10 / physical_constants['atomic unit of length'][0]
# from femtoseconds to au
femtosec2au = 1e-15 / physical_constants['atomic unit of time'][0]
# hartrees to electronvolts
h2ev = physical_constants['Hartree energy in eV'][0]
r2meV = 13605.698  # conversion from rydberg to meV
fs_to_cm = 33356.40952  # conversion from fs to cm-1
fs_to_nm = 299.79246  # conversion from fs to nm
# planck constant in eV * fs
hbar = 1e15 * physical_constants['Planck constant over 2 pi in eV s'][0]

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


def hardness(s: str):
    d = {
        'h': 6.4299, 'he': 12.5449, 'li': 2.3746, 'be': 3.4968, 'b': 4.619, 'c': 5.7410, 'n': 6.8624, 'o': 7.9854,
        'f': 9.1065, 'ne': 10.2303, 'na': 2.4441, 'mg': 3.0146, 'al': 3.5849, 'si': 4.1551, 'p': 4.7258,
        's': 5.2960, 'cl': 5.8662, 'ar': 6.4366, 'k': 2.3273, 'ca': 2.7587, 'sc': 2.8582, 'ti': 2.9578,
        'v': 3.0573, 'cr': 3.1567, 'mn': 3.2564, 'fe': 3.3559, 'co': 3.4556, 'ni': 3.555, 'cu': 3.6544,
        'zn': 3.7542, 'ga': 4.1855, 'ge': 4.6166, 'as': 5.0662, 'se': 5.4795, 'br': 5.9111, 'kr': 6.3418,
        'rb': 2.1204, 'sr': 2.5374, 'y': 2.6335, 'zr': 2.7297, 'nb': 2.8260, 'mo': 2.9221, 'tc': 3.0184,
        'ru': 3.1146, 'rh': 3.2107, 'pd': 3.3069, 'ag': 3.4032, 'cd': 3.4994, 'in': 3.9164, 'sn': 4.3332,
        'sb': 4.7501, 'te': 5.167, 'i': 5.5839, 'xe': 6.0009, 'cs': 0.6829, 'ba': 0.9201, 'la': 1.1571,
        'ce': 1.3943, 'pr': 1.6315, 'nd': 1.8686, 'pm': 2.1056, 'sm': 2.3427, 'eu': 2.5798, 'gd': 2.8170,
        'tb': 3.0540, 'dy': 3.2912, 'ho': 3.5283, 'er': 3.7655, 'tm': 4.0026, 'yb': 4.2395, 'lu': 4.4766,
        'hf': 4.7065, 'ta': 4.9508, 'w': 5.1879, 're': 5.4256, 'os': 5.6619, 'ir': 5.900, 'pt': 6.1367,
        'au': 6.3741, 'hg': 6.6103, 'tl': 1.7043, 'pb': 1.9435, 'bi': 2.1785, 'po': 2.4158, 'at': 2.6528,
        'rn': 2.8899, 'fr': 0.9882, 'ra': 1.2819, 'ac': 1.3497, 'th': 1.4175, 'pa': 1.9368, 'u': 2.2305,
        'np': 2.5241, 'pu': 3.0436, 'am': 3.4169, 'cm': 3.4050, 'bk': 3.9244, 'cf': 4.2181, 'es': 4.5116,
        'fm': 4.8051, 'md': 5.0100, 'no': 5.3926, 'lr': 5.4607}
    return d[s] / 27.211


def xc(s: str) -> Dict:
    d = {'pbe':  {'type':'pure', 'alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.0, 'beta1': 0.2, 'beta2': 1.83},
         'blyp': {'type':'pure','alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.0, 'beta1': 0.2, 'beta2': 1.83},
         'bp':   {'type':'pure','alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.0, 'beta1': 0.2, 'beta2': 1.83},
         'pbe0': {'type':'hybrid','alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.25, 'beta1': 0.2, 'beta2': 1.83},
         'b3lyp': {'type':'hybrid','alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.20, 'beta1': 0.2, 'beta2': 1.83},
         'bhlyp': {'type':'hybrid','alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.50, 'beta1': 0.2, 'beta2': 1.83},
         'cam-b3lyp': {'type':'rhs','alpha1': 1.86, 'alpha2': 0.00, 'ax': 0.38, 'beta1': 0.90, 'beta2': 0.0},
         'lc-blyp': {'type':'rhs', 'alpha1': 8.0, 'alpha2': 0.00, 'ax': 0.53, 'beta1': 4.50, 'beta2': 0.0},
         'wb97': {'type':'rhs', 'alpha1': 8.0, 'alpha2': 0.00, 'ax': 0.61, 'beta1': 4.41, 'beta2': 0.0}}
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
                return [f5[path][()] for path in paths_to_prop]
            else:
                return f5[paths_to_prop][()]
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


def tuplesXYZ_to_plams(xs):
    """ Transform a list of namedTuples to a Plams molecule """
    plams_mol = Molecule()
    for at in xs:
        symb = at.symbol
        cs = at.xyz
        plams_mol.add_atom(Atom(symbol=symb, coords=tuple(cs)))

    return plams_mol


def compute_center_of_mass(atoms: List) -> Tuple:
    """
    Compute the center of mass of a molecule
    """
    # Get the masses of the atoms
    symbols = map(lambda at: at.symbol, atoms)
    masses = np.array([getmass(s) for s in symbols])
    total_mass = np.sum(masses)

    # Multiple the mass by the coordinates
    mrs = [getmass(at.symbol) * np.array(at.xyz) for at in atoms]
    xs = np.sum(mrs, axis=0)

    # Center of mass
    cm = xs / total_mass

    return tuple(cm)


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
