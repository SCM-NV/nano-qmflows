
from math import sqrt
from qmworks.hdf5.quantumHDF5 import cp2k2hdf5
from qmworks.parsers.xyzParser import readXYZ

import h5py
import numpy as np
import os


# =================================<>==========================================
from nac.integrals.overlapIntegral import calcMtxOverlapP

from utilsTest import (change_mol_units, create_dict_CGFs)

from multipoleObaraSaika import sab
from nac.integrals.multipoleIntegrals import (calcMultipoleMatrixP,
                                              build_primitives_gaussian)

# =================================<>==========================================
path_hdf5 = 'test_files/test.hdf5'


def test_overlapMultipoles():
    """
    Test if the Obara-Saika scheme implemented in nac/integrals/obaraSaika.pyx
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
    arr = calcMtxOverlapP(mol, cgfsN)
    brr = calcMtxOverlapP_old(mol, cgfsN)

    r = np.sum(arr - brr)

    assert r < 1e-6
# ====================================<>=======================================


def sijContracted(t1, t2):
    """
    Matrix entry calculation between two Contracted Gaussian functions.
    Equivalent to < t1| t2 >.

    :param t1: tuple containing the cartesian coordinates and primitve gauss
    function of the bra.
    :type t1: (xyz, (Coeff, Expo))
    :param t2: tuple containing the cartesian coordinates and primitve gauss
    function of the ket.
    :type t2: (xyz, (Coeff, Expo))
    """
    gs1 = build_primitives_gaussian(t1)
    gs2 = build_primitives_gaussian(t2)

    return sum(sab(g1, g2) for g1 in gs1 for g2 in gs2)


def calcMatrixEntry(xyz_cgfs, ixs):
    """
    Computed each matrix element using an index a tuple containing the
    cartesian coordinates and the primitives gauss functions.

    :param xyz_cgfs: List of tuples containing the cartesian coordinates
    and the primitive gauss functions.
    :type xyz_cgfs: [(xyz, (Coeff, Expo))]
    :param ixs: Index of the matrix entry to calculate.
    :type ixs: (Int, Int)
    :returns: float
    """
    i, j = ixs
    t1 = xyz_cgfs[i]
    t2 = xyz_cgfs[j]
    return sijContracted(t1, t2)


def calcMtxOverlapP_old(atoms, cgfsN):
    """
    Overlap matrix entry calculation between two Contracted Gaussian functions
    """

    return calcMultipoleMatrixP(atoms, cgfsN, calcMatrixEntry=calcMatrixEntry)
