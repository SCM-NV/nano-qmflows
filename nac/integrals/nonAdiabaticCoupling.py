__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
# from multipoleObaraSaika import sab_unfolded
from functools import partial
from multiprocessing import Pool
from nac.integrals.overlapIntegral import sijContracted
from scipy import sparse
from typing import Dict, List, Tuple

import numpy as np

# Numpy type hints
Vector = np.ndarray
Matrix = np.ndarray


# =====================================<>======================================
def calculateCoupling3Points(
        dt,
        mtx_sji_t0, mtx_sij_t0, mtx_sji_t1, mtx_sij_t1):
    """
    Calculate the non-adiabatic interaction matrix using 3 geometries,
    the CGFs for the atoms and molecular orbitals coefficients read
    from a HDF5 File.
    """
    cte = 1.0 / (4.0 * dt)
    return cte * (3 * (mtx_sji_t1 - mtx_sij_t1) + (mtx_sij_t0 - mtx_sji_t0))


def correct_phases(overlaps, mtx_phases, dim):
    """
    Correct the phases of the overlap matrices
    """
    # Reshape phases vector to matrix
    phases_t0, phases_t1, phases_t2 = [
        mtx_phases[i, :].reshape(dim, 1) for i in range(3)]

    # Matrices containing the phases resulting from multipling
    # the phases of state_i * state_j
    mtx_phases_Sji_t0_t1 = np.dot(phases_t0, phases_t1.transpose())
    mtx_phases_Sji_t1_t2 = np.dot(phases_t1, phases_t2.transpose())
    mtx_phases_Sij_t1_t0 = np.transpose(mtx_phases_Sji_t0_t1)
    mtx_phases_Sij_t2_t1 = np.transpose(mtx_phases_Sji_t1_t2)

    return  [Sji * phases for Sji, phases in
             zip(overlaps, [mtx_phases_Sji_t0_t1, mtx_phases_Sij_t1_t0,
                            mtx_phases_Sji_t1_t2, mtx_phases_Sij_t2_t1])]


def compute_overlaps_for_coupling(
        geometries: Tuple, coefficients: Tuple, dictCGFs: Dict,
        trans_mtx: Matrix=None) -> Matrix:
    """
    Compute the Overlap matrices used to compute the couplings

    :parameter geometries: Tuple of molecular geometries.
    :type      geometries: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :parameter coefficients: Tuple of Molecular Orbital coefficients.
    :type      coefficients: (Matrix, Matrix, Matrix)
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],
    AngularMomentum), Primitive = (Coefficient, Exponent)
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :returns: [Matrix] containing the overlaps at different times
    """
    mol0, mol1, mol2 = geometries
    css0, css1, css2 = coefficients

    # Dictionary containing the number of CGFs per atoms
    cgfs_per_atoms = {s: len(dictCGFs[s]) for s in dictCGFs.keys()}

    # Dimension of the square overlap matrix
    dim = sum(cgfs_per_atoms[at[0]] for at in mol0)

    # Atomic orbitals overlap
    suv_0 = calcOverlapMtx(dictCGFs, dim, mol0, mol1)
    suv_0_t = np.transpose(suv_0)
    suv_1 = calcOverlapMtx(dictCGFs, dim, mol1, mol2)
    suv_1_t = np.transpose(suv_1)

    # Convert the transformation matrix to sparse representation
    trans_mtx = sparse.csr_matrix(trans_mtx)

    # Partial application of the first argument
    spherical_fun = partial(calculate_spherical_overlap, trans_mtx)

    # Overlap matrix for different times in Spherical coordinates
    mtx_sji_t0 = spherical_fun(suv_0, css0, css1)
    mtx_sji_t1 = spherical_fun(suv_1, css1, css2)
    mtx_sij_t0 = spherical_fun(suv_0_t, css1, css0)
    mtx_sij_t1 = spherical_fun(suv_1_t, css2, css1)

    return mtx_sji_t0, mtx_sij_t0, mtx_sji_t1, mtx_sij_t1


def calculate_spherical_overlap(trans_mtx: Matrix, suv: Matrix, css0: Matrix,
                                css1: Matrix) -> Matrix:
    """
    Calculate the Overlap Matrix between molecular orbitals at different times.
    """
    if trans_mtx is not None:
        # Overlap in Sphericals using a sparse representation
        transpose = trans_mtx.transpose()
        suv = trans_mtx.dot(sparse.csr_matrix.dot(suv, transpose))

    css0T = np.transpose(css0)

    return np.dot(css0T, np.dot(suv, css1))


def calcOverlapMtx(dictCGFs: Dict, dim: int,
                   mol0: List, mol1: List):
    """
    Parallel calculation of the overlap matrix using the atomic
    basis at two different geometries: R0 and R1.
    """
    fun_overlap = partial(calc_overlap_row, dictCGFs, mol1, dim)
    fun_lookup = partial(lookup_cgf, mol0, dictCGFs)

    with Pool() as p:
        xss = p.map(partial(apply_nested, fun_overlap, fun_lookup),
                    range(dim))

    return np.stack(xss)


def apply_nested(f, g, i):
    return f(*g(i))


def calc_overlap_row(dictCGFs: Dict, mol1: List, dim: int,
                     xyz_atom0: List, cgf_i: Tuple) -> Vector:
    """
    Calculate the k-th row of the overlap integral using
    2 CGFs  and 2 different atomic coordinates.
    This function only computes the upper triangular matrix since for the atomic
    basis, the folliwng condition is fulfill
    <f(t-dt) | f(t) > = <f(t) | f(t - dt) >
    """
    row = np.zeros(dim)
    acc = 0
    for s, xyz_atom1 in mol1:
        cgfs_j = dictCGFs[s]
        nContracted = len(cgfs_j)
        calc_overlap_atom(
            row, xyz_atom0, cgf_i, xyz_atom1, cgfs_j, acc)
        acc += nContracted
    return row


def calc_overlap_atom(row: Vector, xyz_0: List, cgf_i: Tuple, xyz_1: List,
                      cgfs_atom_j: Tuple, acc: int) -> Vector:
    """
    Compute the overlap between the CGF_i of atom0 and all the
    CGFs of atom1
    """
    for j, cgf_j in enumerate(cgfs_atom_j):
        idx = acc + j
        row[idx] = sijContracted((xyz_0, cgf_i), (xyz_1, cgf_j))


def lookup_cgf(atoms: List, dictCGFs: Dict, i: int) -> Tuple:
    """
    Search for CGFs number `i` in the dictCGFs.
    """
    if i == 0:
        # return first CGFs for the first atomic symbol
        xyz = atoms[0][1]
        r = dictCGFs[atoms[0].symbol][0]
        return xyz, r
    else:
        acc = 0
        for s, xyz in atoms:
            length = len(dictCGFs[s])
            acc += length
            n = (acc - 1) // i
            if n != 0:
                index = length - (acc - i)
                break

    t = xyz, dictCGFs[s][index]

    return t
