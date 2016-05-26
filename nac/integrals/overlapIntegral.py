__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============
from functools import partial
from math import floor, sqrt
from multiprocessing import Pool

import numpy as np
from qmworks.utils import concatMap

# ==================> Internal modules <====================
from obaraSaika import  sab

# ==================================<>======================================


def calcMtxOverlapP(atoms, cgfsN):
    """
    Overlap matrix entry calculation between two Contracted Gaussian functions
    """
    xyz_cgfs = concatMap(lambda rs: createTupleXYZ_CGF(*rs), zip(atoms, cgfsN))
    nOrbs = len(xyz_cgfs)
    # Number of non-zero entries of a triangular mtx
    indexes = calcIndexTriang(nOrbs)
    pool = Pool()
    rss = pool.map(partial(calcMatrixEntry, xyz_cgfs), indexes)
    pool.close()
    
    return np.array(list(rss))


def calcMatrixEntry(xyz_cgfs, ixs):
    i, j = ixs
    t1 = xyz_cgfs[i]
    t2 = xyz_cgfs[j]
    return sijContracted(t1, t2)


def sijContracted(t1, t2):
    """
    Matrix entry calculation between two Contracted Gaussian functions
    """
    r1, cgf1 = t1
    r2, cgf2 = t2
    cs1, es1 = cgf1.primitives
    cs2, es2 = cgf2.primitives
    l1, l2 = [x.orbType for x in [cgf1, cgf2]]
    gs1 = [(r1, l1, rs) for rs in zip(cs1, es1)]
    gs2 = [(r2, l2, rs) for rs in zip(cs2, es2)]

    return sum(sab(g1, g2) for g1 in gs1 for g2 in gs2)


def calcOrbType_Components(l, x):
    """
    Functions related to the orbital momenta indexes
    """
    return orbitalIndexes[l, x]

orbitalIndexes = {
    ("S", 0): 0, ("S", 1): 0, ("S", 2): 0,
    ("Px", 0): 1, ("Px", 1): 0, ("Px", 2): 0,
    ("Py", 0): 0, ("Py", 1): 1, ("Py", 2): 0,
    ("Pz", 0): 0, ("Pz", 1): 0, ("Pz", 2): 1,
    ("Dxx", 0): 2, ("Dxx", 1): 0, ("Dxx", 2): 0,
    ("Dxy", 0): 1, ("Dxy", 1): 1, ("Dxy", 2): 0,
    ("Dxz", 0): 1, ("Dxz", 1): 0, ("Dxz", 2): 1,
    ("Dyy", 0): 0, ("Dyy", 1): 2, ("Dyy", 2): 0,
    ("Dyz", 0): 0, ("Dyz", 1): 1, ("Dyz", 2): 1,
    ("Dzz", 0): 0, ("Dzz", 1): 0, ("Dzz", 2): 2,
    ("Fxxx", 0): 3, ("Fxxx", 1): 0, ("Fxxx", 2): 0,
    ("Fxxy", 0): 2, ("Fxxy", 1): 1, ("Fxxy", 2): 0,
    ("Fxxz", 0): 2, ("Fxxz", 1): 0, ("Fxxz", 2): 1,
    ("Fxyy", 0): 1, ("Fxyy", 1): 2, ("Fxyy", 2): 0,
    ("Fxyz", 0): 1, ("Fxyz", 1): 1, ("Fxyz", 2): 1,
    ("Fxzz", 0): 1, ("Fxzz", 1): 0, ("Fxzz", 2): 2,
    ("Fyyy", 0): 0, ("Fyyy", 1): 3, ("Fyyy", 2): 0,
    ("Fyyz", 0): 0, ("Fyyz", 1): 2, ("Fyyz", 2): 1,
    ("Fyzz", 0): 0, ("Fyzz", 1): 1, ("Fyzz", 2): 2,
    ("Fzzz", 0): 0, ("Fzzz", 1): 0, ("Fzzz", 2): 3
}

def vecNorm(rs):
    return np.linalg.norm(rs)


# ============================================
# Miscelaneus functions


def dimTriang(dim):
    p = int(floor(sqrt(1 + 8 * dim)))
    return (-1 + p) // 2


def calcIndexTriang(n):
    flatDim = (n ** 2 + n) // 2
    xss = np.dstack(np.triu_indices(n))
    return np.reshape(xss, (flatDim, 2))


def createTupleXYZ_CGF(atom, cgfs):
    xyz = atom.xyz
    return [(xyz, cs) for cs in cgfs]
