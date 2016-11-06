from nac.common import InputKey
from os.path import join
from qmworks.hdf5.quantumHDF5 import turbomole2hdf5
from utilsTest import try_to_remove

import h5py

# ===============================<>============================================
path_basis = 'test/test_files/basis_turbomole'
path_hdf5 = 'test/test_files/test.hdf5'
path_MO = 'test/test_files/aomix_ethylene.in'
path_xyz = 'test/test_files/ethylene_au.xyz'

# Path to Nodes in the HDF5 file
path_ethylene = join('/turbomole', 'test', 'ethylene')
path_es = join(path_ethylene, 'eigenvalues')
path_css = join(path_ethylene, 'coefficients')

# Orbital info
number_of_orbs = 36
number_of_orb_funs = 38


def dump_MOs_coeff(handle_hdf5, path_es, path_css, number_of_orbs,
                   number_of_orb_funs):
    """
    MO coefficients are stored in row-major order, they must be transposed
    to get the standard MO matrix.
    :param files: Files to calculate the MO coefficients
    :type  files: Namedtuple (fileXYZ,fileInput,fileOutput)
    :param job: Output File
    :type  job: String
    """
    key = InputKey('orbitals', [path_MO, number_of_orbs, number_of_orb_funs,
                                path_es, path_css])

    turbomole2hdf5(handle_hdf5, [key])

    return path_es, path_css


@try_to_remove([path_hdf5])
def test_store_basisSet():
    """
    Check if the turbomole basis set are read
    and store in HDF5 format.
    """
    keyBasis = InputKey("basis", [path_basis])
    with h5py.File(path_hdf5, chunks=True) as f5:
        turbomole2hdf5(f5, [keyBasis])
        if not f5["turbomole/basis"]:
            assert False
