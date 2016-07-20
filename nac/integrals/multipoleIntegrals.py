
# ==========> Standard libraries and third-party <===============

from multiprocessing import Pool

# ==================> Internal modules <====================
from .overlapIntegral import build_primitives_gaussian
from multipoleObaraSaika import sab_efg

# ==================================<>======================================


def dipoleContracted(t1, t2, rc, e=1, f=1, g=1):
    """
    Matrix entry calculation between two Contracted Gaussian functions.
    Equivalent to < t1| t2 >
    :param t1: tuple containing the cartesian coordinates and primitve gauss
    function of the bra.
    :type t1: (xyz, (Coeff, Expo))
    :param t2: tuple containing the cartesian coordinates and primitve gauss
    function of the ket.
    :type t2: (xyz, (Coeff, Expo))
    :param rc: Cartesian Coordinates where the multipole is centered
    :type rc: Tuple
    """
    gs1 = build_primitives_gaussian(t1)
    gs2 = build_primitives_gaussian(t2)

    return sum(sab_efg(g1, g2, rc, e, f, g) for g1 in gs1 for g2 in gs2)

