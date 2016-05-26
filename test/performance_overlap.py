

from qmworks.hdf5.quantumHDF5 import cp2k2hdf5
from nac.integrals.overlapIntegral import calcMtxOverlapP
from qmworks.parsers.xyzParser import readXYZ

import h5py
import numpy as np
import os

# ========================<>=======================
from utilsTest import (change_mol_units, create_dict_CGFs, triang2mtx, try_to_remove)
# ========================<>=======================

path_hdf5 = 'test_files/test.hdf5'
path_xyz = 'test_files/ethylene.xyz'
path_MO = 'test_files/MO_cartesian.out'


def test_overlap():
    """
    The overlap matrix must fulfill the following equation C^(+) S C = I
    where S is the overlap matrix, C is the MO matrix and
    C^(+) conjugated complex.
    """
    basis = 'DZVP-MOLOPT-SR-GTH'
    # nOrbitals = 46
    # mol = readXYZ('test_files/ethylene_au.xyz')
    mol = readXYZ('test_files/Cd16Se13_6HCOO.xyz')
    mol = change_mol_units(mol)
    labels = [at.symbol for at in mol]

    pathBasis = os.environ['BASISCP2K']

    with h5py.File(path_hdf5, chunks=True) as f5:
        dictCGFs = create_dict_CGFs(f5, cp2k2hdf5, pathBasis,
                                    basis, 'cp2k', mol)

    cgfsN = [dictCGFs[l] for l in labels]
    dim = sum(len(xs) for xs in cgfsN)
    # mtx = triang2mtx(calcMtxOverlapP(mol, cgfsN), dim)
    mtx = calcMtxOverlapP(mol, cgfsN)
    try_to_remove(path_hdf5)
    # print(mtx[0])
    
if __name__ == "__main__":
    test_overlap()
