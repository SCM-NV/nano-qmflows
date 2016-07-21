__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============

from multipoleObaraSaika import sab
from nac.integrals.multipoleIntegrals import (CalcMultipoleMatrixP,
                                              build_primitives_gaussian)
# ====================================<>=======================================


class CalcMtxOverlapP(CalcMultipoleMatrixP):
    """
    Overlap matrix entry calculation between two Contracted Gaussian functions
    """

    def __init__(atoms, cgfsN):
        super.__init__(atoms, cgfsN)

    def calcMatrixEntry(self, xyz_cgfs, ixs):
        """
        Computed each matrix element using an index a tuple containing the
        cartesian coordinates and the primitives gauss functions.
        :param xyz_cgfs: List of tuples containing the cartesian coordinates
        and the primitive gauss functions.
        :type xyz_cgfs: [(xyz, (Coeff, Expo))]
        """
        i, j = ixs
        t1 = xyz_cgfs[i]
        t2 = xyz_cgfs[j]
        return sijContracted(t1, t2)


def sijContracted(t1, t2):
    """
    Matrix entry calculation between two Contracted Gaussian functions.
    Equivalent to < t1| t2 >
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
