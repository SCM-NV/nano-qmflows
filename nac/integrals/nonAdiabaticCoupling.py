__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
# from multipoleObaraSaika import sab_unfolded
from functools import partial
from multiprocessing import Pool
from nac.common import retrieve_hdf5_data
from nac.integrals.overlapIntegral import sijContracted
from scipy import sparse
from typing import Dict, List, Tuple

import numpy as np

# Numpy type hints
Vector = np.ndarray
Matrix = np.ndarray
Tensor3D = np.ndarray

# =====================================<>======================================


def calculate_couplings_levine(i: int, dt: float, w_jk: Matrix,
                               w_kj: Matrix) -> Matrix:
    """
    Compute the non-adiabatic coupling according to:
    `Evaluation of the Time-Derivative Coupling for Accurate Electronic
    State Transition Probabilities from Numerical Simulations`.
    Garrett A. Meek and Benjamin G. Levine.
    dx.doi.org/10.1021/jz5009449 | J. Phys. Chem. Lett. 2014, 5, 2351−2356
    """
    # Orthonormalize the Overlap matrices
    w_jk = np.linalg.qr(w_jk)[0]
    w_kj = np.linalg.qr(w_kj)[0]

    # Diagonal matrix
    w_jj = np.diag(np.diag(w_jk))
    w_kk = np.diag(np.diag(w_kj))

    # remove the values from the diagonal
    np.fill_diagonal(w_jk, 0)
    np.fill_diagonal(w_kj, 0)

    # Components A + B
    acos_w_jj = np.arccos(w_jj)
    asin_w_jk = np.arcsin(w_jk)

    a = acos_w_jj - asin_w_jk
    b = acos_w_jj + asin_w_jk
    A = - np.sin(np.sinc(a))
    B = np.sin(np.sinc(b))

    # Components C + D
    acos_w_kk = np.arccos(w_kk)
    asin_w_kj = np.arcsin(w_kj)

    c = acos_w_kk - asin_w_kj
    d = acos_w_kk + asin_w_kj
    C = np.sin(np.sinc(c))
    D = np.sin(np.sinc(d))

    # Components E
    w_lj = np.sqrt(1 - (w_jj ** 2) - (w_kj ** 2))
    w_lk = -(w_jk * w_jj + w_kk * w_kj) / w_lj
    asin_w_lj = np.arcsin(w_lj)
    asin_w_lk = np.arcsin(w_lk)
    asin_w_lj2 = asin_w_lj ** 2
    asin_w_lk2 = asin_w_lk ** 2

    t1 = w_lj * w_lk * asin_w_lj
    x1 = np.sqrt((1 - w_lj ** 2) * (1 - w_lk ** 2)) - 1
    t2 = x1 * asin_w_lk
    t = t1 + t2
    E_nonzero = 2 * asin_w_lj * t / (asin_w_lj2 - asin_w_lk2)

    # Check whether w_lj is different of zero
    E1 = np.where(np.abs(w_lj) > 1e-8, E_nonzero, np.zeros(A.shape))

    E = np.where(np.isclose(asin_w_lj2, asin_w_lk2), w_lj ** 2, E1)

    cte = 1 / (2 * dt)
    return cte * (np.arccos(w_jj) * (A + B) + np.arcsin(w_kj) * (C + D) + E)


def correct_phases(overlaps: Tensor3D, mtx_phases: Matrix) -> List:
    """
    Correct the phases for all the overlaps
    """
    nFrames = overlaps.shape[0]  # total number of overlap matrices

    for k in (nFrames // 2):
        m = 2 * k
        # Extract phases
        phases_t0, phases_t1 = mtx_phases[k: k + 2]
        mtx_phases_Sji_t0_t1 = np.outer(phases_t0, phases_t1)
        mtx_phases_Sij_t1_t0 = np.transpose(mtx_phases_Sji_t0_t1)

        # Update array with the fixed phases
        overlaps[m] *= mtx_phases_Sji_t0_t1
        overlaps[m + 1] *= mtx_phases_Sij_t1_t0

    return overlaps


def compute_overlaps_for_coupling(
        geometries: Tuple, path_hdf5: str,
        mo_paths: Tuple, dictCGFs: Dict,
        nHOMO: int, couplings_range: Tuple,
        hdf5_trans_mtx: str=None) -> Matrix:
    """
    Compute the Overlap matrices used to compute the couplings

    :parameter geometries: Tuple of molecular geometries.
    :type      geometries: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :parameter mo_paths: Path to the MO inside the HDF5.
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],
    AngularMomentum), Primitive = (Coefficient, Exponent)
    :param trans_mtx: path to the transformation matrix to
    translate from Cartesian to Sphericals.
    :returns: [Matrix] containing the overlaps at different times
    """
    mol0, mol1 = geometries

    # Dictionary containing the number of CGFs per atoms
    cgfs_per_atoms = {s: len(dictCGFs[s]) for s in dictCGFs.keys()}

    # Dimension of the square overlap matrix
    dim = sum(cgfs_per_atoms[at[0]] for at in mol0)

    # Atomic orbitals overlap
    suv_0 = calcOverlapMtx(dictCGFs, dim, mol0, mol1)
    suv_0_t = np.transpose(suv_0)

    css0, css1, trans_mtx = read_overlap_data(
        path_hdf5, mo_paths, hdf5_trans_mtx, nHOMO, couplings_range)

    # Convert the transformation matrix to sparse representation
    trans_mtx = sparse.csr_matrix(trans_mtx)

    # Partial application of the first argument
    spherical_fun = partial(calculate_spherical_overlap, trans_mtx)

    # Overlap matrix for different times in Spherical coordinates
    mtx_sji_t0 = spherical_fun(suv_0, css0, css1)
    mtx_sij_t0 = spherical_fun(suv_0_t, css1, css0)

    return mtx_sji_t0, mtx_sij_t0


def read_overlap_data(path_hdf5, mo_paths, hdf5_trans_mtx, nHOMO, couplings_range):
    """
    Read the Molecular orbital coefficients and the transformation matrix
    """
    mos = retrieve_hdf5_data(path_hdf5, mo_paths)

    # Extract a subset of molecular orbitals to compute the coupling
    lowest, highest = compute_range_orbitals(mos[0], nHOMO, couplings_range)
    css0, css1 = tuple(map(lambda xs: xs[:, lowest: highest], mos))

    # Read the transformation matrix to convert from Cartesian to
    # Spherical coordinates
    if hdf5_trans_mtx is not None:
        trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx)

    return css0, css1, trans_mtx


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

def compute_range_orbitals(mtx: Matrix, nHOMO: int,
                           couplings_range: Tuple) -> Tuple:
    """
    Compute the lowest and highest index used to extract
    a subset of Columns from the MOs
    """
    # If the user does not define the number of HOMOs and LUMOs
    # assume that the first half of the read MO from the HDF5
    # are HOMOs and the last Half are LUMOs.
    _, nOrbitals = mtx.shape
    nHOMO = nHOMO if nHOMO is not None else nOrbitals // 2

    # If the couplings_range variable is not define I assume
    # that the number of LUMOs is equal to the HOMOs.
    if all(x is not None for x in [nHOMO, couplings_range]):
        lowest = nHOMO - couplings_range[0]
        highest = nHOMO + couplings_range[1]
    else:
        lowest = 0
        highest = nOrbitals

    return lowest, highest
