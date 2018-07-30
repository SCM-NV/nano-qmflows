
__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from functools import reduce
from os.path import join
from pymonad import curry

# ==================> Internal modules <==========
from nac.common import (CGF, InputKey)
from qmflows.hdf5.quantumHDF5 import cp2k2hdf5, turbomole2hdf5
from qmflows.utils import (concat, concatMap)
# ==========================<>=================================


def createUniqueCGF(f5, basis_name, package_name, ls):
    """
    Using a HDF5 file Object, it reads the basis set and generates
    the set of unique CGFs for the given atoms
    :param f5:        HDF5 file
    :type  f5:        H5py handler
    :param basis_name: Name of the basis set
    :type  basis_name: String
    :param package_name:  Name of the used software
    :type  package_name:  String
    :param ls:        List of Atomics labels
    :type  ls:        [Strings]
    """
    uniqLabels = set(ls)
    bss = [readBasisSet(f5, basis_name, package_name, l) for l in uniqLabels]

    return (uniqLabels, bss)


def readBasisSet(f5, basis_name, package_name, l):
    """
    :param f5:        HDF5 file
    :type  f5:        H5py handler
    :param basis_name: Name of the basis set
    :type  basis_name: String
    :param package_name:  Name of the used software
    :type  package_name:  String
    :param l:         List of Atomics labels
    :type  l:         Strings
    """
    basis_name = basis_name.upper()
    path_expo = join('/', package_name, "basis", l.lower(), basis_name,
                     "exponents")
    path_coef = join('/', package_name, "basis", l.lower(), basis_name,
                     "coefficients")

    if path_expo not in f5 or path_coef not in f5:
        msg = ('The basis set named:{} is not stored in the HDF5 \
        file'.format(basis_name))
        raise RuntimeError(msg)
    ess = f5[path_expo].value
    dsetCoef = f5[path_coef]
    css = dsetCoef.value
    formatB = dsetCoef.attrs["basisFormat"]

    return generateCGF(ess, css, formatB, package_name)


def generateCGF(ess, css, formats, softName):

    orbLabels = ['s', 'p', 'd', 'f']

    snd = lambda xs: xs[1]

    def fun1(es, css, fs):
        """
        For the Cp2K basis set there is only one set of exponents
        for each coefficients The Format for Cp2k basis is
        [nQuanPrincipal lmin lmax nContrac nS nP nD nF]
        """
        fs = str2list(fs)
        contract = list(zip(fs[4:], orbLabels))

        def go(t1, t2):
            """
            Cp2k Basis have exactly the same number of contracts
            for all angular quantum number (i.e S, P, D, Z) and all
            the same exponent. Using this fact the contracts representation
            is built sharing the exponents between all the contracts
            """
            index, acc = t1
            n, l = t2
            xss = css[index: n + index]
            rss = concatMap(expandBasis_cp2k(l, es), xss)

            return (index + n, acc + rss)

        return snd(reduce(go, contract, (0, [])))

    def fun2(ess, css, fs):

        def funAcc(acc, x):
            n, xs = acc
            return (n + x, xs + [n + x])

        def accum(l, n, xs):
            def go(t, k):
                index, acc = t
                xss = css[index: k + index]
                yss = ess[index: k + index]
                rss = expandBasis_turbomole(l, yss, xss)

                return (index + k, acc + rss)

            return reduce(go, xs, (n, []))

        # print(fs)
        fss = str2ListofList(fs)
        # snd . foldl' funAcc (0,[0]) (map sum fss)
        lens = snd(reduce(funAcc, list(map(sum, fss)), (0, [0])))
        return concat([snd(accum(l, n, xs))
                       for (l, n, xs) in zip(orbLabels, lens, fss)])

    if softName == "cp2k":
        return fun1(ess, css, formats)
    if softName == "turbomole":
        return fun2(ess, css, formats)
    else:
        msg = ("Basis set expansion it is not available for the \
        package: {}".format(softName))
        return NotImplementedError(msg)


@curry
def expandBasis_cp2k(l, es, cs):

    primitives = (cs, es)
    if l == 's':
        return [CGF(primitives, 'S')]
    elif l == 'p':
        return [CGF(primitives, 'Px'), CGF(primitives, 'Py'),
                CGF(primitives, 'Pz')]
    elif l == 'd':
        return [CGF(primitives, 'Dxx'), CGF(primitives, 'Dxy'),
                CGF(primitives, 'Dxz'), CGF(primitives, 'Dyy'),
                CGF(primitives, 'Dyz'), CGF(primitives, 'Dzz')]
    elif l == 'f':
        return [CGF(primitives, 'Fxxx'), CGF(primitives, 'Fxxy'),
                CGF(primitives, 'Fxxz'), CGF(primitives, 'Fxyy'),
                CGF(primitives, 'Fxyz'), CGF(primitives, 'Fxzz'),
                CGF(primitives, 'Fyyy'), CGF(primitives, 'Fyyz'),
                CGF(primitives, 'Fyzz'), CGF(primitives, 'Fzzz')]


@curry
def expandBasis_turbomole(l, es, cs):

    primitives = (cs, es)
    if l == 's':
        return [CGF(primitives, 'S')]
    elif l == 'p':
        return [CGF(primitives, 'Px'), CGF(primitives, 'Py'),
                CGF(primitives, 'Pz')]
    elif l == 'd':
        return [CGF(primitives, 'Dxx'), CGF(primitives, 'Dyy'),
                CGF(primitives, 'Dzz'), CGF(primitives, 'Dxy'),
                CGF(primitives, 'Dxz'), CGF(primitives, 'Dyz')]
    elif l == 'f':
        return [CGF(primitives, 'Fxxx'), CGF(primitives, 'Fyyy'),
                CGF(primitives, 'Fzzz'), CGF(primitives, 'Fxyy'),
                CGF(primitives, 'Fxxy'), CGF(primitives, 'Fxxz'),
                CGF(primitives, 'Fxzz'), CGF(primitives, 'Fyzz'),
                CGF(primitives, 'Fyyz'), CGF(primitives, 'Fxyz')]
    else:
        msg = ("The basis set expansion for this angular momentum has not \
        been implemented yet")
        raise NotImplementedError(msg)


def expandBasisOneCGF(l, es, cs):

    primitives = (cs, es)
    if l == 's':
        return CGF(primitives, 'S')
    elif l == 'p':
        return CGF(primitives, 'Px')
    elif l == 'd':
        return CGF(primitives, 'Dxx')
    elif l == 'f':
        return CGF(primitives, 'Fxxx')


def saveBasis(f5, path_basis, softName):

    keyBasis = InputKey("basis", [path_basis])

    if softName == 'cp2k':
        cp2k2hdf5(f5, [keyBasis])

    elif softName == 'turbomole':
        turbomole2hdf5(f5, [keyBasis])


def str2ListofList(s):
    """
    transform a string to a integer list of lists
    >>>
       str2ListofList("[[1,2,3],[4, 5], [1]")
        [[1,2,3],[4,5],[1]]
    """
    strs = s.replace('[', '').split('],')
    return [list(map(int, xs.replace(']', '').split(','))) for xs in strs]


def str2list(xs):
    """
    read a string like an integer list
    """
    s = xs.replace('[', '').split(']')[0]
    return list(map(int, s.split(',')))
