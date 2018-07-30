__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <============================
from nac.common import (Matrix, Vector)
from os.path import join
from qmflows.utils import concat
from typing import (Dict, List)
import numpy as np
# ==================================<>=========================================


def calc_transf_matrix(
        f5, mol: List, basis_name: str, dict_global_norms: Dict,
        package_name: str) -> Matrix:
    """
    Compute the transformation matrix using the values of Appendix A, from:
    `International Journal of Quantum Chemistry, Vol. 90, 227–243 (2002)`
    """
    symbols = [at.symbol for at in mol]
    uniqSymbols = set(symbols)
    formats = {}
    for elem in uniqSymbols:
        dset = f5[join(package_name, 'basis', elem,
                       basis_name.upper(), "coefficients")]
        formats[elem] = dset.attrs["basisFormat"]
    dict_basisFormat = {elem: read_basis_format(package_name, fs)
                        for elem, fs in formats.items()}
    return build_coeff_matrix(
        dict_basisFormat, symbols, dict_global_norms, package_name)


def build_coeff_matrix(
        dict_basisFormat: Dict, symbols: List, dict_global_norms: Dict,
        package_name: str) -> Matrix:
    """
    using the atomic `symbols` and the basis for a given package,
    compute the transformation matrix.
    :parameter dict_basisFormats: Format containing information
                                  about the Contracted GF
    :type dict_basisFormat: Key= String, Val = [Int] | [[Int]]
    :parameter symbols: Atomics symbols
    :parameter Uniqsymbols: There is only one symbol for
    atom type in the molecule.
    :parameter package_name: Quantum package name.
    :returns: transformation matrix
    """
    # Compute the global norm for all the spherical CGfs
    # Label of the spherical CGFs
    dict_orbital_SLabels = {elem: calc_orbital_Slabels(package_name, fs)
                            for elem, fs in dict_basisFormat.items()}

    # Label of the Cartesian CGFs
    dict_orbital_CLabels = {elem: calc_orbital_Clabels(package_name, fs)
                            for elem, fs in dict_basisFormat.items()}

    # Total number of spherical CGFs
    nSphericals = sum(sum(len(x) for x in dict_orbital_SLabels[el])
                      for el in symbols)
    # Total number of cartesian CGFs
    nCartesians = sum(sum(len(x) for x in dict_orbital_CLabels[el])
                      for el in symbols)

    # Resulting coefficients matrix
    css = np.zeros((nSphericals, nCartesians))

    i, j = 0, 0
    for el in symbols:
        # Retrieve the spherical and Cartesian labels of the CGFs
        slabels = dict_orbital_SLabels[el]
        clabels = dict_orbital_CLabels[el]

        # downcase the atomic label
        el = el.lower()

        # Fill the transformation matrix rows
        k = 0
        for lss, lcs in zip(slabels, clabels):
            len_c = len(lcs)
            for s in lss:
                norm = dict_global_norms[el][k][1]
                rs = calc_transf_per_primitive(
                    s, lcs, dict_orbital_CLabels, dict_orbital_SLabels,
                    dict_coeff_transf)
                css[i, j: j + len(lcs)] = rs * norm
                # Update the index of the row
                i += 1
                k += 1
            # update the indices of the columns
            j += len_c
    return css


def calc_transf_per_primitive(
        slabel: str, clabels: List, dict_orbital_CLabels: Dict,
        dict_orbital_SLabels: dict, dict_coeff_transf: Dict) -> Vector:
    """
    Compute the coefficients to transform from Cartesian to sphericals
    """
    l, m = dict_Slabel_to_lm[slabel]
    cs = []
    for cl in clabels:
        lx, ly, lz = dict_Clabel_to_xyz[cl]
        r = dict_coeff_transf.get((l, m, lx, ly, lz))
        r = r if r is not None else 0
        cs.append(r)
    return np.array(cs)


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
    Labels of the Cartesian CGFs
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
    elif name == 'turbomole':
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
    'p': ['Px', 'Py', 'Pz'],
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
    'px': [1, 1], 'py': [1, -1], 'pz': [1, 0],   # Is these the Cp2k Standard?
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

c_3_4 = np.sqrt(3 / 4)    # 0.866025
c_6_5 = np.sqrt(6 / 5)    # 1.095445
c_3_8 = np.sqrt(3 / 8)    # 0.612372
c_5_8 = np.sqrt(5 / 8)    # 0.790569
c_9_8 = np.sqrt(9 / 8)    # 1.060660
c_9_20 = np.sqrt(9 / 20)  # 0.670820
c_3_40 = np.sqrt(3 / 40)  # 0.273861

dict_coeff_transf = {
    (0, 0, 0, 0, 0): 1,          # S
    (1, -1, 0, 1, 0): 1,         # Py
    (1, 0, 0, 0, 1): 1,          # Pz
    (1, 1, 1, 0, 0): 1,          # Px
    (2, -2, 1, 1, 0): 1,         # Dxy
    (2, -1, 0, 1, 1): 1,         # Dyz
    (2, 0, 2, 0, 0): -0.5,       # Dxx
    (2, 0, 0, 2, 0): -0.5,       # Dyy
    (2, 0, 0, 0, 2): 1,          # Dzz
    (2, 1, 1, 0, 1): 1,          # Dxz
    (2, 2, 2, 0, 0): c_3_4,      # Dxx
    (2, 2, 0, 2, 0): -c_3_4,     # Dyy
    (3, -3, 0, 3, 0): -c_5_8,    # Fy3
    (3, -3, 2, 1, 0): c_9_8,     # Fx2y
    (3, -2, 1, 1, 1): 1,         # Fxyz
    (3, -1, 0, 1, 2): c_6_5,     # Fyz2
    (3, -1, 0, 3, 0): -c_3_8,    # Fy3
    (3, -1, 2, 1, 0): -c_3_40,   # Fx2y
    (3, 0, 0, 0, 3): 1,          # Fz3
    (3, 0, 0, 2, 1): -c_9_20,    # Fy2z
    (3, 0, 2, 0, 1): -c_9_20,    # Fx2z
    (3, 1, 1, 0, 2): c_6_5,      # Fxz2
    (3, 1, 3, 0, 0): -c_3_8,     # Fx3
    (3, 1, 1, 2, 0): -c_3_40,    # Fxy2
    (3, 2, 2, 0, 1): c_3_4,      # Fx2z
    (3, 2, 0, 2, 1): -c_3_4,     # Fy2z
    (3, 3, 3, 0, 0): c_5_8,      # Fx3
    (3, 3, 1, 2, 0): -c_9_8,     # Fxy2
}
