__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============

from multipoleObaraSaika import sab
from nac.integrals.multipoleIntegrals import (general_multipole_matrix,
                                              build_primitives_gaussian)
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


def calcMtxOverlapP(atoms, cgfsN):
    """
    Overlap matrix entry calculation between two Contracted Gaussian functions
    """

    return general_multipole_matrix(atoms, cgfsN, calculator=calcMatrixEntry)
