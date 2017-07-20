__all__ = ['compute_normalization_sphericals', 'create_dict_CGFs',
           'create_normalized_CGFs']

# ================> Python Standard  and third-party <==========
from .contractedGFs import createUniqueCGF
from itertools import (chain, groupby)
from math import pi, sqrt
from nac.common import (CGF, InputKey, product)
from nac.integrals.overlapIntegral import sijContracted
from nac.integrals.multipoleIntegrals import calcOrbType_Components
from os.path import join
from qmworks.utils import chunksOf
from qmworks.hdf5.quantumHDF5 import (cp2k2hdf5, turbomole2hdf5)
from typing import (Dict, List, Tuple)
import numpy as np
import h5py

# =======================> Basis set normalization <===========================
# indices from Spherical to cartesians (Index, coefficients)
#     0      1       2      3        4       5        6     7       8        9
# ['Fxxx', 'Fxxy', 'Fxxz', 'Fxyy', 'Fxyz', 'Fxzz', 'Fyyy', 'Fyyz', 'Fyzz', 'Fzzz']
c_3_4 = np.sqrt(3 / 4)    # 0.866025
c_6_5 = np.sqrt(6 / 5)    # 1.095445
c_3_8 = np.sqrt(3 / 8)    # 0.612372
c_5_8 = np.sqrt(5 / 8)    # 0.790569
c_9_8 = np.sqrt(9 / 8)    # 1.060660
c_9_20 = np.sqrt(9 / 20)  # 0.670820
c_3_40 = np.sqrt(3 / 40)  # 0.273861

dict_spherical_cartesian = {
    'S': [(0, 1)],                                   # S => S
    'Py': [(1, 1)],                                  # Py => Py
    'Pz': [(2, 1)],                                  # Pz => Pz
    'Px': [(0, 1)],                                  # Px => Px
    'D-2': [(1, 1)],                                 # D-2 => Dxy
    'D-1': [(4, 1)],                                 # D-1 => Dyz
    'D0': [(5, 1), (0, -0.5), (3, -0.5)],            # D0 => Dzz - 0.5Dxx - 0.5Dyy
    'D+1': [(2, 1)],                                 # D-1 => Dxz
    'D+2': [(0, c_3_4), (3, -c_3_4)],                # D-2 => Dxx - Dyy
    'F-3': [(6, -c_5_8), (1, c_9_8)],                # F-3 => Fx2y - fy3
    'F-2': [(4, 1)],                                 # F-2 => Fxyz
    'F-1': [(8, c_6_5), (6, -c_3_8), (1, -c_3_40)],  # F-1 => Fyz2 - Fy3 - Fx2y
    'F0': [(9, 1), (7, -c_9_20), (2, -c_9_20)],      # F0 => Fz3 - Fy2z - Fx2z
    'F+1': [(5, c_6_5), (0, -c_3_8), (3, -c_3_40)],  # F+1 => Fxz2 -Fx3 - Fxy2
    'F+2': [(2, c_3_4), (7, -c_3_4)],                # F+2 => Fx2z - Fy2z
    'F+3': [(0, c_5_8), (3, -c_9_8)]                 # F+3 => Fx3 - Fxy2
}


def compute_normalization_sphericals(dictCGFs: Dict) -> Dict:
    """
    Use the global normalization for the CGFs that constitute the sphericals.
    Taken from the following paper:
    'Mutual Conversion of Three Flavors of Gaussian Type Orbitals'.
    International Journal of Quantum Chemistry, Vol. 90, 227–243 (2002)

    :returns: dict containing the normalization constant for each CGF in spher
    """
    return {l: compute_normalization_cgfs(cgfs)
            for l, cgfs in dictCGFs.items()}


def compute_normalization_cgfs(cgfs: List) -> List:
    """
    Compute the Normalization constant of the whole CGFs for an atom
    in spherical coordinates.
    """
    # Number of CGFs for each angularM
    CGFs_cartesians = {'S': 1, 'P': 3, 'D': 6, 'F': 10}

    norms = [
        [compute_normalizations(g, chunk)
         for chunk in chunksOf(list(vals), CGFs_cartesians[g])]
        for g, vals in groupby(cgfs, lambda x: x.orbType[0])]

    return list(chain(*chain(*norms)))


def compute_normalizations(label: str, cgfs: List) -> List:
    """
    compute the normalization constant for the spherical of the same angular
    momentum.
    """
    spherical_labels = {
        'S': ['S'], 'P': ['Py', 'Pz', 'Px'],
        'D': ['D-2', 'D-1', 'D0', 'D+1', 'D+2'],
        'F': ['F-3', 'F-2', 'F-1', 'F0', 'F+1', 'F+2', 'F+3']}

    # Retrieve the labels of the CGFs components  in sphericals
    ang_labels = spherical_labels[label]

    return list(map(
        lambda l: compute_normalization_per_cgf(cgfs, l), ang_labels))


def compute_normalization_per_cgf(cgfs: List, label: str) -> List:
    """
    Compute the normalization contanst of a single spherical CGF.
    Using the coefficients reported at:
    `International Journal of Quantum Chemistry, Vol. 90, 227–243 (2002)`
    """
    def compute_norm(ti: Tuple, tj: Tuple) -> float:
        # Get the index coefficient pairs
        i, ci = ti
        j, cj = tj
        # Retrieve the corresponding CGF
        cgfi = cgfs[i]
        cgfj = cgfs[j]

        # Integrate at zero
        r = [0, 0, 0]
        sij = ci * cj * sijContracted((r, cgfi), (r, cgfj))

        return sij

    # Retrieve the linear combination of Cartesian the compose
    # the spherical CGF
    cgfs_indices = dict_spherical_cartesian[label]

    #  compute the norm of the components
    norm2 = sum(sum(
        compute_norm(ti, tj) for tj in cgfs_indices) for ti in cgfs_indices)

    # returns the global norm of the CGF
    global_norm = np.sqrt(1 / norm2)

    return label, global_norm


def create_dict_CGFs(
        path_hdf5: dict, basis_name: str, mol: List, package_name='cp2k',
        package_config: Dict=None):
    """
    Try to read the basis from the HDF5 otherwise read it from a file and store
    it in the HDF5 file. Finally, it reads the basis Set from HDF5 and
    calculate the CGF for each atom.

    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    type path_hdf5: String
    :param basisname: Name of the Gaussian basis set.
    :type basisname: String
    :param xyz: List of Atoms.
    :type xyz: [nac.common.AtomXYZ]
    """
    functions = {'cp2k': cp2k2hdf5, 'turbomole': turbomole2hdf5}

    basis_location = join(package_name, 'basis')
    with h5py.File(path_hdf5) as f5:
        if basis_location not in f5:
            # Search Path to the file containing the basis set
            pathBasis = package_config["basis"]
            keyBasis = InputKey("basis", [pathBasis])
            # Store the basis sets
            functions[package_name](f5, [keyBasis])

        # CGFs in Cartesian coordinates
        cgfs = create_normalized_CGFs(f5, basis_name, package_name, mol)

    return cgfs


def create_normalized_CGFs(f5, basis_name, package_name: str, mol: List) -> Dict:
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
    :param basis_name: Name of the basis set
    :param package_name:  Name of the used software
    :param mol: list of tuples containing the atomic label and
    cartesian coordinates
    :returns: dict containing the CGFs
    """
    ls = [atom.symbol for atom in mol]
    # create only one set of CGF for each atom in the molecule
    (uniqls, uniqCGFs) = createUniqueCGF(f5, basis_name, package_name, ls)

    uniqCGFsN = [list(map(norm_primitive, cgfs)) for cgfs in uniqCGFs]

    return {l: cgf for (l, cgf) in zip(uniqls, uniqCGFsN)}


def norm_primitive(cgf, r=None):
    """
    Calculate global normalization constant.
    """
    l = cgf.orbType
    es = cgf.primitives[1]
    csN = normCoeff(cgf)

    return CGF((csN, es), l)


def normCoeff(cgf):
    cs, es = cgf.primitives
    orbType = cgf.orbType
    indices = [calcOrbType_Components(orbType, k) for k in range(3)]
    newCs = [normalize_primitive(c, e, indices) for (c, e) in zip(cs, es)]
    return np.array(newCs)


def normalize_primitive(c: float, e: float, indices: List) -> float:
    """
    Normalize a primitive Gaussian function
    """
    n = (sqrt(ang_fun(e, indices) * (pi / (2.0 * e)) ** 1.5))

    return c / n


def ang_fun(e: float, indices: List) -> float:
    """
    compute the angular product
    """
    prod = product(facOdd(2 * k - 1) for k in indices)
    return prod / (4 * e) ** sum(indices)

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
