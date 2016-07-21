
# ==========> Standard libraries and third-party <===============
from functools import partial
from multiprocessing import Pool
from qmworks.utils import concatMap

import numpy as np
# ==================> Internal modules <====================
from .overlapIntegral import build_primitives_gaussian
from .multipoleObaraSaika import sab, sab_efg
from .overlapIntegral import calcIndexTriang, createTupleXYZ_CGF

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


def calcMatrixEntry(xyz_cgfs, ixs):
    """
    Computed each matrix element using an index a tuple containing the
    cartesian coordinates and the primitives gauss functions.
    :param xyz_cgfs: List of tuples containing the cartesian coordinates and
    the primitive gauss functions
    :type xyz_cgfs: [(xyz, (Coeff, Expo))]
    """
    i, j = ixs
    t1 = xyz_cgfs[i]
    t2 = xyz_cgfs[j]
    return sijContracted(t1, t2)


def calculateDipoleIntegrals(atoms, cgfsN):
    """
    """
    xyz_cgfs = concatMap(lambda rs: createTupleXYZ_CGF(*rs),
                         zip(atoms, cgfsN))
    nOrbs = len(xyz_cgfs)
    # Number of non-zero entries of a triangular mtx
    indexes = calcIndexTriang(nOrbs)
    pool = Pool()
    rss = pool.map(partial(calcMatrixEntry, xyz_cgfs), indexes)
    pool.close()

    return np.array(list(rss))


def triang2mtx(arr, dim):
    rss = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            k = fromIndex([i, j], [dim, dim])
            rss[i, j] = arr[k]
    return rss


def calculateDipoleCenter(atoms, cgfsN):
    """
    """
    overlap = calcMtxOverlapP(atoms, cgfsN)
    
