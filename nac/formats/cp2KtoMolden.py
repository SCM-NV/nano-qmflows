
__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from functools import partial, reduce
from pymonad import curry
from os.path import join


# ==================> Internal modules <==========
from nac.basisSet.basisNormalization import cp2kCoefftoMoldenFormat
from nac.common import getmass
from qmworks.utils import flatten, snd, zipWith, zipWith3

# ===========================>HDF5 Functions<===================================


def writeMoldenFormat(output, f5, atoms, basisName, occup):

    header = "[Molden Format]\n[Title]\n\n"
    index = list(range(1, 1 + len(atoms)))
    basisName = basisName.upper()
    xyzAtoms = "[Atoms] AU\n" + flatten([writeAtomSection(*rs)
                                         for rs in zip(atoms, index)])
    basis = "[GTO]\n" + flatten(writeBasisSection(f5, basisName, *rs)
                                for rs in zip(atoms, index))
    coeffs = "[MO]\n" + flatten(writeMOSection(f5, occup))
    with open(output, "w") as out:
        out.write(header + xyzAtoms + basis + coeffs)


def writeAtomSection(atom, i):
    l = atom.label
    xyz = atom.xyz
    x, y, z = xyz[0], xyz[1], xyz[2]
    f = ' {0:>2}{1:5d}{2:5d} {3: .14E} {4: .14E} {5: .14E}\n'
    st = f.format(l, i, getmass(l), x, y, z)

    return st


def writeBasisSection(f5, basisSet, atom, i):
    l = atom.label
    pathExpo = join("/cp2k/basis", l, basisSet, "exponents")
    pathCoef = join("/cp2k/basis", l, basisSet, "coefficients")
    dsetExpo = f5[pathExpo]
    dsetCoef = f5[pathCoef]
    es = dsetExpo[...]
    css = dsetCoef[...]
    formatB = dsetCoef.attrs["basisFormat"]
    newCss = cp2kCoefftoMoldenFormat(es, css, formatB)
    contract = flatten(writeContraction(es, newCss, formatB))

    return '{:>5d} 0\n{}\n'.format(i, contract)


def writeMOSection(f5, occup):
    pathEs = "/cp2k/mo/eigenvalues"
    pathCs = "/cp2k/mo/coefficients"
    dsetEs = f5[pathEs]
    dsetCs = f5[pathCs]
    es = dsetEs[...]
    css = dsetCs[...]
    index = list(range(1, 1 + len(es)))

    return zipWith3(writeMO(occup))(es)(css)(index)

# ========> Functions to format the Basis Set <============


def writeContraction(es, css, fs):
    orbS = ['s', 'p', 'd', 'f']
    nContrac = int(fs[3])
    contract = list(zip(fs[4:], orbS))

    def go(acc_i, nl):
        index, acc = acc_i
        n, l = nl
        xss = css[index: n + index]
        funC = partial(writeCoeff, nContrac, es, l)
        rss = list(map(funC, xss))
        return (index + n, acc + flatten(rss))

    return snd(reduce(go, contract, (0, '')))


def writeCoeff(nc, es, label, cs):

    header = '{:<s}{:>5d} 1.00\n'.format(label, nc)
    funP = lambda x, y: '{:.14E} {: .14E}\n'.format(x, y)
    pairs = zipWith(funP)(es)(cs)

    return header + flatten(pairs)


# def orbitals2Molden(outout, filehdf5, l):
#     path = join("/cp2k/basis", l, basisSet.upper())
#     dset = filehdf5(path)
#     fs = dset.attrs["basisformat"]
#     css = filehdf5[key]
#     return css


@curry
def writeMO(occup, e, cs, i):
    """
    Functions to Format the MO
    """

    oc = 0 if (i > occup) else 2
    f = " Sym=    {:d}a\n Ene={: .14E}\n Spin= Alpha\n Occup={: 9.6f}\n"
    header = f.format(i, e, oc)
    funMO = lambda x, i: '{:>5d} {: .14E}\n'.format(i, x)
    index = list(range(1, 1 + len(cs)))
    mo = flatten([funMO(*xs) for xs in zip(iter(cs), index)])

    return header + mo

