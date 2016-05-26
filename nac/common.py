
__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from collections import namedtuple
import h5py

# ======================================================================
# Named Tuples
AtomData       = namedtuple("AtomData", ("label", "coordinates", "cgfs"))
AtomBasisKey   = namedtuple("AtomBasisKey", ("atom", "basis", "basisFormat"))
AtomBasisData  = namedtuple("AtomBasisData", ("exponents", "coefficients"))
AtomXYZ        = namedtuple("AtomXYZ", ("symbol", "xyz"))
CGF            = namedtuple("CGF", ("primitives", "orbType"))
InfoMO         = namedtuple("InfoMO", ("eigenVals", "coeffs"))
InputKey       = namedtuple("InpuKey", ("name", "args"))
MO             = namedtuple("MO", ("coordinates", "cgfs", "coefficients"))


def getmass(s):
    d = {'h':1,'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb' : 37, 'sr' : 38 , 'Y' : 39, 'zr' : 40, 'nb' : 41, 'mo' : 42, 'tc' : 43, 'ru' : 44, 'rh' : 45, 'pd' : 46, 'ag' : 47, 'cd' : 48 , 'in' : 49, 'sn' : 50, 'sb' : 51, 'te' : 52, 'i' : 53, 'xe' : 54, 'cs' : 55}

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
                return  [f5[path][...] for path in paths_to_prop]
            else:
                return f5[paths_to_prop][...]
    except KeyError:
        msg = "There is not {} stored in the HDF5\n".format(paths_to_prop)
        with open('cp2k_out', 'a') as f:
            f.write(msg)
        raise KeyError(msg)
    except FileNotFoundError:
        msg = "there is not HDF5 file containing the numerical results"
        raise RuntimeError(msg)
