
__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from itertools import groupby
from functools import partial, reduce
from math import pi, sqrt
from pymonad import curry

import numpy as np

# =======================> Internal modules <==================================
from .contractedGFs import createUniqueCGF, expandBasisOneCGF
from nac.common import AtomData, CGF
from nac.integrals.overlapIntegral import sijContracted
from nac.integrals.multipoleIntegrals import calcOrbType_Components

from qmworks.utils import concatMap, fst, product, snd

# =======================> Basis set normalization <===========================


def normalizeCGFAtoms(atoms):
    """
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
    """
    def fun(atom):
        s = atom.symbol
        xyz = atom.coordinates
        cgfs = atom.cgfs
        cgfsN = [normGlobal(cs, xyz) for cs in cgfs]
        return AtomData(s, xyz, cgfsN)

    return list(map(fun, atoms))


def createNormalizedCGFs(f5, basisName, softName, mol):
    """
    Using a HDF5 file object, it reads the basis set and generates
    the set of CGF for the given stored as a Plams Molecule
    :param f5:        HDF5 file
    :type  f5:        h5py handler
    :param basisName: Name of the basis set
    :type  basisName: String
    :param softName:  Name of the used software
    :type  softName:  String
    :param atomXYZ: list of tuples containing the atomic label and
    cartesian coordinates
    :type  atomXYZ: list of named Tuples ATomXYZ
    """
    ls = [atom.symbol for atom in mol]
    # create only one set of CGF for each atom in the molecule
    (uniqls, uniqCGFs) = createUniqueCGF(f5, basisName, softName, ls)

    uniqCGFsN = [list(map(normGlobal, cgfs)) for cgfs in uniqCGFs]

    return {l: cgf for (l, cgf) in zip(uniqls, uniqCGFsN)}


def normGlobal(cgf, r=[0.0] * 3):
    l = cgf.orbType
    cs, es = cgf.primitives
    csN = normCoeff(cgf)
    cgfN = CGF((csN, es), l)
    sij = sijContracted((r, cgfN), (r, cgfN))
    n = sqrt(1.0 / sij)
    newCs = [n * x for x in csN]

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


def cgftoMoldenFormat(cgf):
    cs, es = cgf.primitives
    l = cgf.orbType
    r0 = [0.0] * 3
    csN = normCoeff(cgf)
    cgfN = CGF((csN, es), l)
    sij = sijContracted((r0, cgfN), (r0, cgfN))
    n = sqrt(1.0 / sij)
    newCs = [n * x for x in cs]

    return newCs


# Software dependent functions
# =============================>Cp2K<=================================


def cp2kCoefftoMoldenFormat(es, css, formatB):
    orbLabels = ['s', 'p', 'd', 'f']
    fsCont = list(map(int, formatB[4:]))
    contract = list(zip(fsCont, orbLabels))

    def go(acc_i, nl):
        index, acc = acc_i
        n, l = nl
        xss = css[index: n + index]
        funCGF = partial(expandBasisOneCGF, l, es)
        rss = list(map(funCGF, xss))
        return (index + n, acc + rss)

    cgfs = snd(reduce(go, contract, (0, [])))

    return np.array(list(map(cgftoMoldenFormat, cgfs)))


# ===========================>Turbomole<=======================================


def turbomoleBasis2MoldenFormat(file_h5, pathBasis, basisName, softName, ls):

    def funMolden(cgf):
        cs, es = cgf.primitives
        l = cgf.orbType
        newCS = cgftoMoldenFormat(cgf)
        return CGF((newCS, es), l)

    atomsCGF = createUniqueCGF(file_h5, basisName, softName, ls)
    cgfsNormal = [list(map(funMolden, cgfs)) for cgfs in atomsCGF]

    return cgfsNormal


def getOneCoeff(cgfs):

    @curry
    def pred(l, cgf):
        l2 = cgf.orbType
        return l == l2

    s = list(filter(pred('S'), cgfs))
    p = list(filter(pred('Px'), cgfs))
    d = list(filter(pred('Dxx'), cgfs))
    f = list(filter(pred('Fxxx'), cgfs))
    return concatMap(lambda xs: [fst(x.primitives) for x in xs],
                     [s, p, d, f])

# ============================================
# Auxiliar functions


def groupByOrbType(cgfs):
    def funOrb(cs):
        l = cs.orbType
        if l[0] == 'S':
            return 's'
        elif l[0] == 'Px':
            return 'p'
        elif l[0] == 'Dxx':
            return 'd'
        elif l[0] == 'Fxxx':
            return 'f'
    return groupby(cgfs, funOrb)


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
