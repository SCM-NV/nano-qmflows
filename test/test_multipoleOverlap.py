
from math import sqrt
from qmworks.hdf5.quantumHDF5 import cp2k2hdf5
from qmworks.parsers.xyzParser import readXYZ

import h5py
import numpy as np
import os


# =================================<>==========================================
from nac.integrals.overlapIntegral import calcMtxOverlapP

from utilsTest import (change_mol_units, create_dict_CGFs, triang2mtx,
                       try_to_remove)

# =================================<>==========================================
path_hdf5 = 'test_files/test.hdf5'


def test_overlapMultipoles():
    """
    Test if the Obara-Saika scheme implemented in in nac/integrals/obaraSaika.pyx
    is the same that the general multipole integrals implemented in
    nac/integrals/multipoleObaraSaika.pyx
    """
    basis = 'DZVP-MOLOPT-SR-GTH'
    mol = readXYZ('test_files/Cd16Se13_6HCOO.xyz')
    mol = change_mol_units(mol)
    labels = [at.symbol for at in mol]

    pathBasis = os.environ['BASISCP2K']

    with h5py.File(path_hdf5) as f5:
        dictCGFs = create_dict_CGFs(f5, cp2k2hdf5, pathBasis,
                                    basis, 'cp2k', mol)
    cgfsN = [dictCGFs[l] for l in labels]
    print(cgfsN)
    arr = calcMtxOverlapP(mol, cgfsN)

    with h5py.File(path_hdf5) as f5:
        f5.require_dataset('overlap_multipole', shape=np.shape(arr), data=arr,
                           dtype=np.float32)

    
