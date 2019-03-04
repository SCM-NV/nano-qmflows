__author__ = "Felipe Zapata"

from qmflows.utils import concat


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
