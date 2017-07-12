
# ==========> Standard libraries and third-party <===============
from functools import partial
from multiprocessing import (cpu_count, Pool)
from nac.common import (Matrix, Vector)

import numpy as np
# ==================> Internal modules <====================
from multipoleObaraSaika import sab_efg  # compiled with cython

from typing import (Callable, Dict, List, Tuple)

# ==================================<>======================================


def general_multipole_matrix(
        molecule: List, dictCGFs: List,
        calculator: Callable=None) -> Vector:
    """
    Generic function to calculate a matrix using a Gaussian basis set and
    the molecular geometry.
    Build a matrix using a pool of worker and a function takes nuclear
    corrdinates and a Contracted Gauss function and compute a number.

    :param molecule: Atomic label and cartesian coordinates.
    :param dictCGFs: Contracted gauss functions normalized, represented as
    a dict of list containing the Contracted Gauss primitives
    :param calculator: Function to compute the matrix elements.
    :returns: Numpy Array representing a flatten triangular matrix.
    """
    # Indices of the cartesian coordinates and corresponding CGFs
    indices, nOrbs = compute_CGFs_indices(molecule, dictCGFs)

    # Create a list of indices of a triangular matrix to distribute
    # the computation of the matrix uniformly among the available CPUs
    block_triang_indices = compute_block_triang_indices(nOrbs)
    with Pool() as p:
        rss = p.map(partial(calculator, molecule, dictCGFs, indices),
                    block_triang_indices)

    return np.concatenate(rss)


def dipoleContracted(
        t1: Tuple, t2: Tuple, rc: Tuple, e: int, f: int, g: int):
    """
    Matrix entry calculation between two Contracted Gaussian functions.
    Equivalent to < t1| t2 >.

    :param t1: tuple containing the cartesian coordinates and primitve gauss
    function of the bra.
    :type t1: (xyz, (Coeff, Expo))
    :param t2: tuple containing the cartesian coordinates and primitve gauss
    function of the ket.
    :type t2: (xyz, (Coeff, Expo))
    :param rc: Cartesian Coordinates where the multipole is centered
    :type rc: Tuple
    :returns: Float
    """
    gs1 = build_primitives_gaussian(t1)
    gs2 = build_primitives_gaussian(t2)

    return sum(sab_efg(g1, g2, rc, e, f, g) for g1 in gs1 for g2 in gs2)


def calcMatrixEntry(
        rc: Tuple, e: int, f: int, g: int, molecule: List, dictCGFs: Dict,
        indices_cgfs: Matrix, indices_triang: Matrix) -> Vector:
    """
    Computed each matrix element using an index a tuple containing the
    cartesian coordinates and the primitives gauss functions.

    :param rc: Multipole center
    :type rc: (Float, Float, Float)
    :param xyz_cgfs: List of tuples containing the cartesian coordinates and
    the primitive gauss functions
    :type xyz_cgfs: [(xyz, (Coeff, Expo))]
    :param ixs: Index of the matrix entry to calculate.
    :type ixs: (Int, Int)
    :returns: float
    """
    # Number of total orbitals
    result = np.empty(indices_triang.shape[0])

    for k, (i, j) in enumerate(indices_triang):
        # Extract contracted and atom indices
        at_i, cgfs_i_idx = indices_cgfs[i]
        at_j, cgfs_j_idx = indices_cgfs[j]

        # Extract atom
        atom_i = molecule[at_i]
        atom_j = molecule[at_j]
        # Extract CGFs
        cgf_i = dictCGFs[atom_i.symbol.lower()][cgfs_i_idx]
        cgf_j = dictCGFs[atom_j.symbol.lower()][cgfs_j_idx]

        # Contracted Gauss functions and nuclear coordinates
        ti = atom_i.xyz, cgf_i
        tj = atom_j.xyz, cgf_j
        result[k] = dipoleContracted(ti, tj, rc, e, f, g)

    return result


def calcMtxMultipoleP(atoms: List, dictCGFs: Dict, rc=(0, 0, 0), e=0, f=0, g=0):
    """
    Multipole matrix entry calculation between two Contracted Gaussian functions.
    It uses a partial applied function to pass the center of the multipole `rc`
    and the coefficients of the operator x^e y^f z^g.

    :param atoms: Atomic label and cartesian coordinates
    :param cgfsN: Contracted gauss functions normalized, represented as
    a dictionary list of tuples of coefficients and Exponents.
    :param calcMatrixEntry: Function to compute the matrix elements.
    :type calcMatrixEntry: Function
    :param rc: Multipole center
    :type rc: (Float, Float, Float)
    :returns: Numpy Array representing a flatten triangular matrix.
    """
    curriedFun = partial(calcMatrixEntry, rc, e, f, g)

    return general_multipole_matrix(atoms, dictCGFs, calculator=curriedFun)


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
    Functions related to the orbital momenta indexes.

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
    """
    Calculate the indexes of the matrix that is represented as a flatten array.
    """
    flatDim = (n ** 2 + n) // 2
    xss = np.dstack(np.triu_indices(n))
    return np.reshape(xss, (flatDim, 2))


def createTupleXYZ_CGF(atom, cgfs):
    xyz = atom.xyz
    return map(lambda cs: (xyz, cs), cgfs)


def compute_CGFs_indices(mol: List, dictCGFs: Dict) -> Tuple:
    """
    Create a matrix of indices of dimension nOrbs x 2.
    Where the first column contains the index atom and the second
    the index of the CGFs relative to the atom
    """
    # List of the length for each atom in the molecule
    lens = [len(dictCGFs[at.symbol.lower()]) for at in mol]
    nOrbs = sum(lens)

    # Array containing the index of both atoms and CGFs
    indices = np.empty((nOrbs, 2), dtype=np.int32)

    acc = 0
    for i, at in enumerate(mol):
        nContracted = lens[i]
        slices  = acc + nContracted
        # indices of the CGFs
        indices[acc: slices, 1] = np.arange(nContracted)
        # index of the atom
        indices[acc: slices, 0] = i
        acc += nContracted

    return indices, nOrbs


def compute_block_triang_indices(nOrbs: int) -> List:
    """
    Create the list of indices of the triangular matrix to be
    distribute approximately uniform among the available CPUs.
    """
    # Indices of the triangular matrix
    indices = np.stack(np.triu_indices(nOrbs), axis=1)

    # Number of entries in a triangular matrix
    dim_triang = indices.shape[0]

    # Available CPUs
    nCPUs = cpu_count()

    # Number of entries to compute for each CPU
    chunk = dim_triang // nCPUs

    # Remaining entries
    rest = dim_triang % nCPUs

    xs = []
    acc = 0
    for i in range(nCPUs):
        b = 1 if i < rest else 0
        upper = acc + chunk + b
        xs.append(indices[acc: upper])
        acc = upper

    return xs
