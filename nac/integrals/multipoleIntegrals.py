import numpy as np

# ==================> Internal modules <====================
from .overlapIntegral import build_primitives_gaussian

# ==================================<>======================================

def dipoleContracted(t1, t2):
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

    return sum(multipole(g1, g2, order=1) for g1 in gs1 for g2 in gs2)


def multipole(g1, g2, order=1):
    """
    Multipole integrals calculated by the Obara-Saika method. See:
    Molecular Electronic-Structure Theory. T. Helgaker, P. Jorgensen, J. Olsen. 
    John Wiley & Sons. 2000, pages: 346-347. 
    :param g1: primitive Gaussian representing the bra
    :type g1: (Coeff, Expo)
    :param g2: primitive Gaussian representing the ket
    :type g2: (Coeff, Expo)
    """
