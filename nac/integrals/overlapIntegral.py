__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============

from multipoleObaraSaika import sab
from nac.common import (Matrix, Vector)
from nac.integrals.multipoleIntegrals import (
    build_primitives_gaussian, general_multipole_matrix)
from typing import (Dict, List, Tuple)
import numpy as np

# ====================================<>=======================================


def sijContracted(t1: Tuple, t2: Tuple):
    """
    Matrix entry calculation between two Contracted Gaussian functions.
    Equivalent to < t1| t2 >.

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


def calc_overlap_triang(
        molecule: List, dictCGFs: Dict, indices_cgfs: Matrix,
        indices_triang: Matrix) -> Vector:
    """
    Compute the upper triangular overlap matrix
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
        result[k] = sijContracted(ti, tj)

    return result


def calcMtxOverlapP(molecule: List, dictCGFs: Dict, runner='multiprocessing') -> Vector:
    """
    Overlap matrix entries calculated using Contracted Gaussian
    functions.

    :returns: flatten upper triangular matrix
    """
    return general_multipole_matrix(
        molecule, dictCGFs, runner=runner, calculator=calc_overlap_triang)
