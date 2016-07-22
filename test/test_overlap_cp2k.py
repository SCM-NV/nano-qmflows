
from qmworks.common import InputKey
from qmworks.hdf5.quantumHDF5 import cp2k2hdf5
from qmworks.integrals.overlapIntegral import calcMtxOverlapP
from qmworks.parsers.xyzParser import readXYZ

from os.path import join
import h5py
import numpy as np
import os

# ======================================<>=====================================
from utilsTest import  (create_dict_CGFs, dump_MOs_coeff,
                        offdiagonalTolerance, triang2mtx, try_to_remove)
# ======================================<>=====================================

path_hdf5 = 'test_files/test.hdf5'
path_xyz = 'test_files/ethylene.xyz'
path_MO = 'test_files/MO_cartesian.out'


def test_store_basisSet():
    """
    """
    pathBasis = os.environ['BASISCP2K']
    keyBasis = InputKey("basis", [pathBasis])
    try_to_remove(path_hdf5)
    with h5py.File(path_hdf5, chunks=True) as f5:
        try:
            # Store the basis sets
            cp2k2hdf5(f5, [keyBasis])
            os.remove(path_hdf5)
            try_to_remove(path_hdf5)
        except RuntimeError:
            try_to_remove(path_hdf5)
            assert False


def test_store_MO_h5():
    """
    test if the MO are stored in the HDF5 format
    """
    path = join('/cp2k', 'test', 'ethylene')
    pathEs = join(path, 'eigenvalues')
    pathCs = join(path, 'coefficients')
    nOrbitals = 46

    try_to_remove(path_hdf5)
    with h5py.File(path_hdf5, chunks=True) as f5:
        pathEs, pathCs = dump_MOs_coeff(f5, cp2k2hdf5, path_MO, pathEs, pathCs, 
                                        nOrbitals)
        if f5[pathEs] and f5[pathCs]:
            try_to_remove(path_hdf5)
            assert True
        else:
            try_to_remove(path_hdf5)
            assert False


# def test_overlap():
#     """
#     The overlap matrix must fulfill the following equation C^(+) S C = I
#     where S is the overlap matrix, C is the MO matrix and
#     C^(+) conjugated complex.
#     """
#     basis = 'DZVP-MOLOPT-SR-GTH'
#     # nOrbitals = 46
#     mol = readXYZ('test_files/ethylene_au.xyz')
#     labels = [at.symbol for at in mol]

#     path = join('/cp2k', 'test', 'ethylene')
#     pathEs = join(path, 'eigenvalues')
#     pathCs = join(path, 'coefficients')
#     pathBasis = os.environ['BASISCP2K']
#     nOrbitals = 46

#     with h5py.File(path_hdf5, chunks=True) as f5:
#         dictCGFs = create_dict_CGFs(f5, cp2k2hdf5, pathBasis, basis, 'cp2k', mol)
#         _, pathCs = dump_MOs_coeff(f5, cp2k2hdf5, path_MO, pathEs, pathCs, nOrbitals)
#         dset = f5[pathCs]
#         trr = dset[...]

#     css = np.transpose(trr)
#     cgfsN = [dictCGFs[l] for l in labels]
#     dim = sum(len(xs) for xs in cgfsN)
#     mtxP = triang2mtx(calcMtxOverlapP(mol, cgfsN), dim)
#     rs = np.dot(trr, np.dot(mtxP, css))

#     # xs = format_aomix(mtxP, dim)
#     # with open("overlap.out", 'w') as f:
#     #     f.write(xs)
#     try_to_remove(path_hdf5)

#     assert offdiagonalTolerance(rs)


