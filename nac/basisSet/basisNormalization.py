
__all__ = ['createNormalizedCGFs']

# ================> Python Standard  and third-party <==========
from math import pi, sqrt

# =======================> Internal modules <==================================
from .contractedGFs import createUniqueCGF
from nac.common import (CGF, product)
from nac.integrals.overlapIntegral import sijContracted
from nac.integrals.multipoleIntegrals import calcOrbType_Components
import numpy as np
# =======================> Basis set normalization <===========================


def createNormalizedCGFs(f5, basisName, softName, mol):
    """
    Using a HDF5 file object, it reads the basis set and generates
    the set of normalized Contracted Gaussian functions.
    The basis set is expanded in contracted gaussian function
    and the normalization.
    The norm of each contracted is given by the following equation
    N = sqrt $ ((2l -1)!! (2m-1)!! (2n-1)!!)/(4*expo)^(l+m+n)  *
        (pi/(2*e))**1.5
    where expo is the exponential factor of the contracted

    let |fi> = sum ci* ni* x^lx * y^ly * z ^lz * exp(-ai * R^2)
    where ni is a normalization constant for each gaussian basis
    then <fi|fj>  = sum sum ci* cj * ni * nj * <Si|Sj>
    where N is the normalization constant
    then the global normalization constant is given by
    N = sqrt (1 / sum sum  ci* cj * ni * nj * <Si|Sj> )
    Therefore the contracted normalized gauss function is given by
    |Fi> = N * (sum ci* ni* x^lx * y^ly * z ^lz * exp(-ai * R^2))

    :param f5:        HDF5 file
    :type  f5:        h5py handler
    :param basisName: Name of the basis set
    :type  basisName: String
    :param softName:  Name of the used software
    :type  softName:  String
    :param mol: list of tuples containing the atomic label and
    cartesian coordinates
    :type  mol: list of named Tuples ATomXYZ
    """
    ls = [atom.symbol for atom in mol]
    # create only one set of CGF for each atom in the molecule
    (uniqls, uniqCGFs) = createUniqueCGF(f5, basisName, softName, ls)

    uniqCGFsN = [list(map(normGlobal, cgfs)) for cgfs in uniqCGFs]

    return {l: cgf for (l, cgf) in zip(uniqls, uniqCGFsN)}


def normGlobal(cgf, r=[0, 0, 0]):
    l = cgf.orbType
    _, es = cgf.primitives
    csN = normCoeff(cgf)
    cgfN = CGF((csN, es), l)
    sij = sijContracted((r, cgfN), (r, cgfN))
    n = sqrt(1.0 / sij)
    # newCs = [n * x for x in csN]
    newCs = n * np.array(csN)

    return CGF((newCs, es), l)


def normCoeff(cgf):
    cs, es = cgf.primitives
    orbType = cgf.orbType
    indexes = [calcOrbType_Components(orbType, k) for k in range(3)]
    prod = product(facOdd(2 * k - 1) for k in indexes)
    angFun = lambda x: prod / (4 * x) ** sum(indexes)
    fun = lambda c, e: c / (sqrt(angFun(e) *
                                 (pi / (2.0 * e)) ** 1.5))
    newCs = [fun(c, e) for (c, e) in zip(cs, es)]
    return newCs


# ============================================
# Auxiliar functions


def facOdd(i):
    """
    (2k -1) !! = (2k)!/(2^k * k!)
    i = 2k - 1 => k = (i + 1)/ 2
    Odd factorial function
    """
    if i == 1:
        return 1
    elif i % 2 == 0:
        msg = 'Factorial Odd function required an odd integer as input'
        raise NameError(msg)
    else:
        k = (1 + i) // 2
        return fac(2 * k) / (2 ** k * fac(k))


def fac(i):
    """
    Factorial function
    """
    if i < 0:
        msg = 'Factorial functions takes natural numbers as argument'
        raise NameError(msg)
    elif i == 0:
        return 1
    else:
        return product(range(1, i + 1))
