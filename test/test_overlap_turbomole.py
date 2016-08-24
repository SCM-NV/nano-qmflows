

from qmworks.hdf5.quantumHDF5 import turbomole2hdf5
from qmworks.parsers.xyzParser import readXYZ

from os.path import join
import h5py
import numpy as np

# ===============================<>============================================
from nac.schedule.components import create_dict_CGFs
from nac.common import InputKey
from nac.integrals.overlapIntegral import calcMtxOverlapP

from utilsTest import (offdiagonalTolerance, triang2mtx, try_to_remove)

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


@try_to_remove([path_hdf5])
def test_store_MO_h5():
    """
    test if the MO are stored in the HDF5 format
    """

    with h5py.File(path_hdf5, chunks=True) as f5:
        path_eigenvals, path_coeffs = dump_MOs_coeff(f5, path_es, path_css,
                                                     number_of_orbs,
                                                     number_of_orb_funs)
        if not(f5[path_eigenvals] and f5[path_coeffs]):
            assert False


@try_to_remove([path_hdf5])
def test_overlap():
    """
    The overlap matrix must fulfill the following equation C^(+) S C = I
    where S is the overlap matrix, C is the MO matrix and
    C^(+) conjugated complex.
    """
    basis = 'def2-SV(P)'
    mol = readXYZ(path_xyz)
    labels = [at.symbol for at in mol]

    # Build the Conctracted Gauss Functions
    dictCGFs = create_dict_CGFs(path_hdf5, basis, mol,
                                package_name='turbomole',
                                package_config={'basis': path_basis})
    cgfsN = [dictCGFs[l] for l in labels]

    with h5py.File(path_hdf5, chunks=True) as f5:
        path_eigenvals, path_coeffs = dump_MOs_coeff(f5, path_es, path_css,
                                                     number_of_orbs,
                                                     number_of_orb_funs)
        trr = f5[path_css].value

    dim = sum(len(xs) for xs in cgfsN)
    css = np.transpose(trr)
    mtx_overlap = triang2mtx(calcMtxOverlapP(mol, cgfsN), dim)
    rs = np.dot(trr, np.dot(mtx_overlap, css))

    assert offdiagonalTolerance(rs)
