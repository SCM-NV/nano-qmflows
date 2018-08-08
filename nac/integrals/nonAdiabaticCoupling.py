__all__ = ['calculate_couplings_3points', 'calculate_couplings_levine',
           'compute_overlaps_for_coupling', 'correct_phases']

# ================> Python Standard  and third-party <==========
from functools import partial
from multiprocessing import cpu_count
from nac.common import (Matrix, Vector, Tensor3D, retrieve_hdf5_data)
from nac.integrals.multipoleIntegrals import (
    compute_CGFs_indices, runner_mpi, runner_multiprocessing)
from nac.integrals.overlapIntegral import sijContracted
from scipy import sparse
from typing import Dict, List, Tuple

import numpy as np


def calculate_couplings_3points(
        dt: float, mtx_sji_t0: Matrix, mtx_sij_t0: Matrix,
        mtx_sji_t1: Matrix, mtx_sij_t1: Matrix) -> None:
    """
    Calculate the non-adiabatic interaction matrix using 3 geometries,
    the CGFs for the atoms and molecular orbitals coefficients read
    from a HDF5 File.
    """
    cte = 1.0 / (4.0 * dt)
    return cte * (3 * (mtx_sji_t1 - mtx_sij_t1) + (mtx_sij_t0 - mtx_sji_t0))


def calculate_couplings_levine(dt: float, w_jk: Matrix,
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
    nOverlaps = overlaps.shape[0]  # total number of overlap matrices
    dim = overlaps.shape[1]  # Size of the square matrix

    for k in range(nOverlaps):
        # Extract phases
        phases_t0, phases_t1 = mtx_phases[k: k + 2]
        phases_t0 = phases_t0.reshape(dim, 1)
        phases_t1 = phases_t1.reshape(1, dim)
        mtx_phases_Sji_t0_t1 = np.dot(phases_t0, phases_t1)

        # Update array with the fixed phases
        overlaps[k] *= mtx_phases_Sji_t0_t1

    return overlaps


def compute_overlaps_for_coupling(
        geometries: Tuple, path_hdf5: str,
        mo_paths: List, dictCGFs: Dict,
        nHOMO: int, couplings_range: Tuple,
        hdf5_trans_mtx: str=None) -> Tuple:
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

    # Atomic orbitals overlap
    suv_0 = calcOverlapMtx(dictCGFs, mol0, mol1)

    css0, css1, trans_mtx = read_overlap_data(
        path_hdf5, mo_paths, hdf5_trans_mtx, nHOMO, couplings_range)

    # Convert the transformation matrix to sparse representation
    trans_mtx = sparse.csr_matrix(trans_mtx)

    # Partial application of the first argument
    spherical_fun = partial(calculate_spherical_overlap, trans_mtx)

    # Overlap matrix for different times in Spherical coordinates
    mtx_sji_t0 = spherical_fun(suv_0, css0, css1)

    return mtx_sji_t0


def read_overlap_data(path_hdf5: str, mo_paths: str, hdf5_trans_mtx: str,
                      nHOMO: int, couplings_range: tuple) -> None:
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


def calcOverlapMtx(
        dictCGFs: Dict, mol0: List, mol1: List,
        runner='multiprocessing', ncores: int=None) -> Matrix:
    """
    Parallel calculation of the overlap matrix using the atomic
    basis at two different geometries: R0 and R1.
    :param mol0: Atomic label and cartesian coordinates of the first geometry.
    :param mol1: Atomic label and cartesian coordinates of the second geometry.
    :param dictCGFs: Contracted gauss functions normalized, represented as
    a dict of list containing the Contracted Gauss primitives
    :param calculator: Function to compute the matrix elements.
    :param runner: function to compute the elements of the matrix
    :param ncores: number of available cores
    """
    # Compute the indices of the nuclear coordinates and CGFs
    # pairs
    indices, nOrbs = compute_CGFs_indices(mol0, dictCGFs)
    partial_fun = partial(calc_overlap_chunk, dictCGFs, mol0, mol1, indices)
    ncores = ncores if ncores is not None else cpu_count()
    
    if runner.lower() == 'mpi':
        result = runner_mpi(partial_fun, nOrbs, ncores)
        return result.reshape(nOrbs, nOrbs)
    else:
        xss = runner_multiprocessing(
            partial_fun, create_rows_range(nOrbs, ncores))

        return np.vstack(xss)


def calc_overlap_chunk(dictCGFs: Dict, mol0: List, mol1: List,
                       indices_cgfs: Matrix, row_slice: Tuple) -> Matrix:
    """
    Compute the row of the overlap matrix indicated by the indexes
    given at row_slice.
    """
    # Indices to compute a subset of the overlap matrix
    lower, upper = row_slice
    nOrbs = indices_cgfs.shape[0]
    chunk_size = upper - lower
    # Matrix containing the partial overlap matrix
    rows = np.empty((chunk_size, nOrbs))

    # Compute the sunset of the overlap matrix
    for k, i in enumerate(range(lower, upper)):
        # Atom and CGFs index
        at_i, cgfs_i_idx = indices_cgfs[i]
        # Extract atom and  CGFs
        atom_i = mol0[at_i]
        cgf_i = dictCGFs[atom_i.symbol.lower()][cgfs_i_idx]
        # Compute the ith row of the overlap matrix
        rows[k] = calc_overlap_row(
            dictCGFs, atom_i.xyz, cgf_i, mol1, indices_cgfs)

    return rows


def calc_overlap_row(dictCGFs: Dict, xyz_0: List, cgf_i: List,
                     mol1: List, indices_cgfs: Matrix) -> Vector:
    """
    Calculate the k-th row of the overlap integral using
    2 CGFs  and 2 different atomic coordinates.
    """
    nOrbs = indices_cgfs.shape[0]
    row = np.empty(nOrbs)

    for k, (at_j, cgfs_j_idx) in enumerate(np.rollaxis(indices_cgfs, axis=0)):
        # Extract atom and  CGFs
        atom_j = mol1[at_j]
        cgf_j = dictCGFs[atom_j.symbol.lower()][cgfs_j_idx]
        xyz_1 = atom_j.xyz
        row[k] = sijContracted((xyz_0, cgf_i), (xyz_1, cgf_j))

    return row


def create_rows_range(nOrbs: int, ncores: int) -> List:
    """
    Create a list of indexes for the row of the overlap matrix
    that will be calculated by a pool of workers.
    """
    # Number of rows to compute for each CPU
    chunk = nOrbs // ncores

    # Remaining entries
    rest = nOrbs % ncores

    xs = []
    acc = 0
    for i in range(ncores):
        b = 1 if i < rest else 0
        upper = acc + chunk + b
        xs.append((acc, upper))
        acc = upper

    return xs
