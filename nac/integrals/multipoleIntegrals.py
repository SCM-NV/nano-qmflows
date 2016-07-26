
# ==========> Standard libraries and third-party <===============
from functools import partial
from multiprocessing import Pool
from qmworks.utils import concatMap

import numpy as np
# ==================> Internal modules <====================
from multipoleObaraSaika import sab_efg  # compiled with cython

# ==================================<>======================================


def calcMultipoleMatrixP(atoms, cgfsN, calcMatrixEntry=None):
    """
    Generic function to calculate a matrix using a Gaussian basis set and
    the molecular geometry.
    :param atoms: Atomic label and cartesian coordinates
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]
    :param calcMatrixEntry: Function to compute the matrix elements.
    :type calcMatrixEntry: Function
    :returns: Numpy Array
    """
    def run(fun_calc_entry):
        """
        Build a matrix using a pool of worker and a function takes nuclear
        corrdinates and a Contracted Gauss function and compute a number.
        """
        xyz_cgfs = concatMap(lambda rs: createTupleXYZ_CGF(*rs),
                             zip(atoms, cgfsN))
        nOrbs = len(xyz_cgfs)
        # Number of non-zero entries of a triangular mtx
        indexes = calcIndexTriang(nOrbs)
        pool = Pool()
        rss = pool.map(partial(fun_calc_entry, xyz_cgfs), indexes)
        pool.close()

        return np.array(list(rss))

    return run(calcMatrixEntry)


def dipoleContracted(t1, t2, rc, e, f, g):
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


def calcMatrixEntry(xyz_cgfs, ixs, rc, e, f, g):
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
    return dipoleContracted(t1, t2, rc, e, f, g)


def calcMtxMultipoleP(atoms, cgfsN, rc, e=0, f=0, g=0):
    """
    Multipole matrix entry calculation between two Contracted Gaussian functions.
    It uses a partial applied function to pass the center of the multipole `rc`
    and the coefficients of the operator x^e y^f z^g
    """
    curriedFun = lambda ts, ixs: calcMatrixEntry(ts, ixs, rc, e, f, g)
    
    return calcMultipoleMatrixP(atoms, cgfsN, calcMatrixEntry=curriedFun)


# ==================================<>=========================================
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


def calcOrbType_Components(l, x):
    """
    Functions related to the orbital momenta indexes
    :param l: Orbital momentum label
    :type l: String
    :param x: cartesian Component (x, y or z)
    :param x: Int
    :returns: integer representing orbital momentum l.
    """
    return orbitalIndexes[l, x]


def build_primitives_gaussian(t):
    """
    Creates a primitve Gaussian function represented by a tuple containing
    the Cartesian coordinates where it is centered, the spin momentum label
    (S, Px, Py, Pz, etc.) and the Coefficients and exponent of it.
    """
    r, cgf = t
    cs, es = cgf.primitives
    l = cgf.orbType
    return list(map(lambda rs: (r, l, rs), zip(cs, es)))


def calcIndexTriang(n):
    flatDim = (n ** 2 + n) // 2
    xss = np.dstack(np.triu_indices(n))
    return np.reshape(xss, (flatDim, 2))


def createTupleXYZ_CGF(atom, cgfs):
    xyz = atom.xyz
    return [(xyz, cs) for cs in cgfs]
    
