__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============
from math import sqrt
from os.path import join

from nac.common import (binomial, even, fac, odd, product)
from qmworks.utils import (concat, concatMap)
import numpy as np
# ==================================<>=========================================


def calc_transf_matrix(f5, mol, basisName, packageName):
    """
    calculate the transformation of the overlap matrix from both spherical
    to cartesian and from cartesian to spherical, see:
    **H. B. Schlegel, M. J. Frisch, Int. J. Quantum Chem. 54, 83 (1995)**
    """
    symbols = [at.symbol for at in mol]
    uniqSymbols = set(symbols)
    formats = {}
    for elem in uniqSymbols:
        dset = f5[join(packageName, 'basis', elem,
                       basisName.upper(), "coefficients")]
        formats[elem] = dset.attrs["basisFormat"]
    dict_basisFormat = {elem: read_basis_format(packageName, fs)
                        for elem, fs in formats.items()}
    return build_coeff_matrix(dict_basisFormat, symbols,
                              uniqSymbols, packageName)


def build_coeff_matrix(dict_basisFormat, symbols, uniqSymbols, packageName):
    """
    Computes equation 15 of
    **H. B. Schlegel, M. J. Frisch, Int. J. Quantum Chem. 54, 83 (1995)**
    :parameter dict_basisFormats: Format containing information
                                  about the Contracted GF
    :type dict_basisFormat: Key= String, Val = [Int] | [[Int]]
    :parameter symbols: Atomics symbols
    :type symbols: [String]
    :parameter Uniqsymbols: There is only one symbol for
    atom type in the molecule
    :type symbols: [String]
    :parameter packName: Quantum package name
    :type packName: string
    """
    dict_orbital_SLabels = {elem: calc_orbital_Slabels(packageName, fs)
                            for elem, fs in dict_basisFormat.items()}

    dict_orbital_CLabels = {elem: calc_orbital_Clabels(packageName, fs)
                            for elem, fs in dict_basisFormat.items()}
    lmax = 3  # Up to f-orbitals
    dict_coeff_transf = calc_dict_spherical_cartesian(lmax)

    print(dict_orbital_SLabels)

    def calc_transf_per_primitive(slabel, clabels):
        l, m = dict_Slabel_to_lm[slabel]
        cs = []
        for cl in clabels:
            lx, ly, lz = dict_Clabel_to_xyz[cl]
            r = dict_coeff_transf.get((l, m, lx, ly, lz))
            r = r if r is not None else 0.0
            cs.append(r)
        return np.array(cs)

    spherical_orbital_labels = concatMap(lambda el:
                                         dict_orbital_SLabels[el], symbols)

    cartesian_orbital_labels = concatMap(lambda el:
                                         dict_orbital_CLabels[el], symbols)

    nSphericals = sum(len(xs) for xs in spherical_orbital_labels)

    nCartesians = sum(len(xs) for xs in cartesian_orbital_labels)

    css = np.zeros((nSphericals, nCartesians))

    i, j = 0, 0
    for (slabels, clabels) in zip(spherical_orbital_labels,
                                  cartesian_orbital_labels):
        len_s = len(slabels)
        len_c = len(clabels)

        for k, s in enumerate(slabels):
            rs = calc_transf_per_primitive(s, clabels)
            css[i + k, j: j + len_c] = rs
        j += len_c
        i += len_s
    return css


def calc_dict_spherical_cartesian(lmax):
    """
    The implementation use a dictionary with key (l,m,lx,ly,lz) to store the
    value of the transformation coefficients.
    :parameter lmax: Total angular momentum
    :type lmax: Int
    """
    d = {}
    for l in range(lmax + 1):
        for lx in range(l + 1):
            for ly in range(l - lx + 1):
                lz = l - lx - ly
                for m in range(-l, l + 1):
                    ma = abs(m)
                    j = (lx + ly - ma)
                    if not(j >= 0 and even(j)):
                        d[(l, m, lx, ly, lz)] = 0
                    else:
                        j = j // 2
                        s1 = 0
                        b = (l - ma) // 2
                        for i in range(b + 1):
                            s2 = 0
                            for k in range(j + 1):
                                s = calc_s_factor(lx, m, ma, k)
                                s2 += binomial(j, k) * \
                                    binomial(ma, lx - 2 * k) * s
                            s1 += calc_s1_factor(l, ma, i, j, s2)
                        root = calc_sqrt(l, ma, lx, ly, lz)
                        d[(l, m, lx, ly, lz)] = root * s1 / (2 ** l * fac(l))

    return d


def calc_s_factor(lx: int, m: int, ma: int, k: int) -> float:
    if (m < 0 and odd(ma - lx)) or (m > 0 and even(ma - lx)):
        e = (ma - lx + 2 * k) // 2
        s = (-1.0) ** e * sqrt(2)
    elif m == 0 and even(lx):
        e = k - lx / 2
        s = (-1.0) ** e
    else:
        s = 0
    return s


def calc_s1_factor(l: int, ma: int, i: int, j: int, s2: float) -> float:
    b1 = binomial(l, i)
    b2 = binomial(i, j)
    return b1 * b2 * (-1) ** i * fac(2 * l - 2 * i) / fac(l - ma - 2 * i) * s2


def calc_sqrt(l: int, ma: int, lx: int, ly: int, lz: int) -> float:
    """
    calculate the square root of equation 15 of:
    **H. B. Schlegel, M. J. Frisch, Int. J. Quantum Chem. 54, 83 (1995)**
    """
    [l2, l2x, l2y, l2z] = [2 * x for x in [l, lx, ly, lz]]
    lmp = l + ma
    lmm = l - ma
    xs = [l, l2, l2x, l2y, l2z, lx, ly, lz, lmp, lmm]
    fl, fl2, fl2x, fl2y, fl2z, flx, fly, flz, flmp, flmm = list(map(fac, xs))
    return sqrt(product([fl2x, fl2y, fl2z, fl, flmm]) /
                product([fl2, flx, fly, flz, flmp]))


def calc_orbital_Slabels(name, fss):
    """
    Most quantum packages use standard basis set which contraction is
    presented usually by a format like:
    c def2-SV(P)
    # c     (7s4p1d) / [3s2p1d]     {511/31/1}
    this mean that this basis set for the Carbon atom uses 7 ``s`` CGF,
    4 ``p`` CGF and 1 ``d`` CGFs that are contracted in 3 groups of 5-1-1
    ``s`` functions, 3-1 ``p`` functions and 1 ``d`` function. Therefore
    the basis set format can be represented by [[5,1,1], [3,1], [1]].

    On the other hand Cp2k uses a special basis set ``MOLOPT`` which
    format explanation can be found at: `C2pk
    <https://github.com/cp2k/cp2k/blob/e392d1509d7623f3ebb6b451dab00d1dceb9a248/cp2k/data/BASIS_MOLOPT>`_.

    :parameter name: Quantum package name
    :type name: string
    :parameter fss: Format basis set
    :type fss: [Int] | [[Int]]
    """
    def funSlabels(d, l, fs):
        if isinstance(fs, list):
            fs = sum(fs)
        labels = [d[l]] * fs
        return labels

    angularM = ['s', 'p', 'd', 'f', 'g']
    if name == 'cp2k':
        dict_Ord_Labels = dict_cp2kOrder_spherical
    else:
        raise NotImplementedError

    return concat([funSlabels(dict_Ord_Labels, l, fs)
                   for l, fs in zip(angularM, fss)])


def calc_orbital_Clabels(name, fss):
    """
    """
    def funClabels(d, l, fs):
        if isinstance(fs, list):
            fs = sum(fs)
        labels = [d[l]] * fs
        return labels

    angularM = ['s', 'p', 'd', 'f', 'g']
    if name == 'cp2k':
        dict_Ord_Labels = dict_cp2kOrd_cartesian
    if name == 'turbomole':
        dict_Ord_Labels = dict_turbomoleOrd_cartesian
        raise NotImplementedError

    return concat([funClabels(dict_Ord_Labels, l, fs)
                  for l, fs in zip(angularM, fss)])


def read_basis_format(name, basisFormat):
    if name == 'cp2k':
        s = basisFormat.replace('[', '').split(']')[0]
        fss = list(map(int, s.split(',')))
        fss = fss[4:]  # cp2k coefficient formats start in column 5
        return fss
    elif name == 'turbomle':
        strs = s.replace('[', '').split('],')
        return [list(map(int, s.replace(']', '').split(','))) for s in strs]
    else:
        raise NotImplementedError()


dict_cp2kOrder_spherical = {
    's': ['s'],
    'p': ['py', 'pz', 'px'],
    'd': ['d-2', 'd-1', 'd0', 'd+1', 'd+2'],
    'f': ['f-3', 'f-2', 'f-1', 'f0', 'f+1', 'f+2', 'f+3']
}

dict_cp2kOrd_cartesian = {
    's': ['S'],
    'p': ['Py', 'Pz', 'Px'],
    'd': ['Dxx', 'Dxy', 'Dxz', 'Dyy', 'Dyz', 'Dzz'],
    'f': ['Fxxx', 'Fxxy', 'Fxxz', 'Fxyy', 'Fxyz', 'Fxzz',
          'Fyyy', 'Fyyz', 'Fyzz', 'Fzzz']
}

dict_turbomoleOrd_cartesian = {
    's': 'S',
    'p': ['Px', 'Py', 'Pz'],
    'd': ['Dxx', 'Dyy', 'Dzz', 'Dxy', 'Dxz', 'Dyz'],
    'f': ['Fxxx', 'Fyyy', 'Fzzz', 'Fxyy', 'Fxxy', 'Fxxz', 'Fxzz',
          'Fyzz', 'Fyyz', 'Fxyz']
}

dict_Slabel_to_lm = {
    's': [0, 0],
    'px': [1, -1], 'py': [1, 0], 'pz': [1, 1],   # Is these the Cp2k Standard?
    'd-2': [2, -2], 'd-1': [2, -1], 'd0': [2, 0], 'd+1': [2, 1], 'd+2': [2, 2],
    'f-3': [3, -3], 'f-2': [3, -2], 'f-1': [3, -1], 'f0': [3, 0],
    'f+1': [3, 1], 'f+2': [3, 2], 'f+3': [3, 3]
}

dict_Clabel_to_xyz = {
    "S": [0, 0, 0],
    "Px": [1, 0, 0],
    "Py": [0, 1, 0],
    "Pz": [0, 0, 1],
    "Dxx": [2, 0, 0],
    "Dxy": [1, 1, 0],
    "Dxz": [1, 0, 1],
    "Dyy": [0, 2, 0],
    "Dyz": [0, 1, 1],
    "Dzz": [0, 0, 2],
    "Fxxx": [3, 0, 0],
    "Fxxy": [2, 1, 0],
    "Fxxz": [2, 0, 1],
    "Fxyy": [1, 2, 0],
    "Fxyz": [1, 1, 1],
    "Fxzz": [1, 0, 2],
    "Fyyy": [0, 3, 0],
    "Fyyz": [0, 2, 1],
    "Fyzz": [0, 1, 2],
    "Fzzz": [0, 0, 3]
}
