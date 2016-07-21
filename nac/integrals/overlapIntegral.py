__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============
from functools import partial
from math import floor, sqrt
from multiprocessing import Pool

import numpy as np
from qmworks.utils import concatMap

# ==================> Internal modules <====================
# from multipoleObaraSaika import sab
from multipoleObaraSaika import sab

# ==================================<>======================================
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


class CalcMultipoleMatrixP:
    """
    Generic class to calculate a matrix using a Gaussian basis set and
    the molecular geometry.
    """
    def __init__(self, atoms, cgfsN):
        """
        :param atoms: Atomic label and cartesian coordinates
        type atoms: List of namedTuples
        :param cgfsN: Contracted gauss functions normalized, represented as a list
        of tuples of coefficients and Exponents.
        type cgfsN: [(Coeff, Expo)]
        """
        self.atoms = atoms
        self.cgfsN = cgfsN

    def __call__(self):
        """
        Build a matrix using a pool of worker and a function takes nuclear
        corrdinates and a Contracted Gauss function and compute a number.
        :returns: Numpy Array
        """
        def calcIndexTriang(n):
            flatDim = (n ** 2 + n) // 2
            xss = np.dstack(np.triu_indices(n))
            return np.reshape(xss, (flatDim, 2))

        xyz_cgfs = concatMap(lambda rs: createTupleXYZ_CGF(*rs),
                             zip(self.atoms, self.cgfsN))
        nOrbs = len(xyz_cgfs)
        # Number of non-zero entries of a triangular mtx
        indexes = calcIndexTriang(nOrbs)
        pool = Pool()
        rss = pool.map(partial(self.calcMatrixEntry, xyz_cgfs), indexes)
        pool.close()

        return np.array(list(rss))

    def calcMatrixEntry(self):
        """
        Function to compute every element of the matrix.
        """
        raise NotImplementedError("The subclass must defined this method")


class CalcOverlapMtx(CalcMultipoleMatrixP):
    """
    Overlap matrix entry calculation between two Contracted Gaussian functions
    """

    def __init__(atoms, cgfsN):
        super.__init__(atoms, cgfsN)

    def calcMatrixEntry(self, xyz_cgfs, ixs):
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
