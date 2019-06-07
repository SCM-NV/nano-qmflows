__all__ = ['calculate_couplings_3points', 'calculate_couplings_levine',
           'compute_overlaps_for_coupling', 'correct_phases']

from compute_integrals import compute_integrals_couplings
from nac.common import (
    Matrix, Tensor3D, retrieve_hdf5_data, tuplesXYZ_to_plams)
from os.path import join
import numpy as np
import os
import uuid


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
    A = - np.sin(np.sinc(a / np.pi))
    B = np.sin(np.sinc(b / np.pi))

    # Components C + D
    acos_w_kk = np.arccos(w_kk)
    asin_w_kj = np.arcsin(w_kj)

    c = acos_w_kk - asin_w_kj
    d = acos_w_kk + asin_w_kj
    C = np.sin(np.sinc(c / np.pi))
    D = np.sin(np.sinc(d / np.pi))

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


def correct_phases(overlaps: Tensor3D, mtx_phases: Matrix) -> list:
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
        config: dict, pair_molecules: tuple, coefficients: tuple) -> tuple:
    """
    Compute the Overlap matrices used to compute the couplings
    :returns: [Matrix] containing the overlaps at different times
    """
    # Atomic orbitals overlap
    suv = calcOverlapMtx(config,  pair_molecules)

    # Read Orbitals Coefficients
    css0, css1 = coefficients

    return np.dot(css0.T, np.dot(suv, css1))


def read_overlap_data(config: dict, mo_paths: list) -> tuple:
    """
    Read the Molecular orbital coefficients and the transformation matrix
    """
    mos = retrieve_hdf5_data(config.path_hdf5, mo_paths)

    # Extract a subset of molecular orbitals to compute the coupling
    lowest, highest = compute_range_orbitals(config)
    css0, css1 = tuple(map(lambda xs: xs[:, lowest: highest], mos))

    return css0, css1


def compute_range_orbitals(config: dict) -> tuple:
    """
    Compute the lowest and highest index used to extract
    a subset of Columns from the MOs
    """
    lowest = config.nHOMO - (config.mo_index_range[0] + config.active_space[0])
    highest = config.nHOMO + config.active_space[1] - config.mo_index_range[0]

    return lowest, highest


def calcOverlapMtx(config: dict, molecules: tuple) -> Matrix:
    """
    Parallel calculation of the overlap matrix using the libint2 library
    at two different geometries: R0 and R1.
    """
    mol_i, mol_j = tuple(tuplesXYZ_to_plams(x) for x in molecules)

    # unique molecular paths
    path_i = join(config["scratch_path"], "molecule_{}.xyz".format(uuid.uuid4()))
    path_j = join(config["scratch_path"], "molecule_{}.xyz".format(uuid.uuid4()))

    # Write the molecules in atomic units
    mol_i.write(path_i)
    mol_j.write(path_j)

    basis_name = config["cp2k_general_settings"]["basis"]
    try:
        integrals = compute_integrals_couplings(path_i, path_j, config["path_hdf5"], basis_name)
    finally:
        os.remove(path_i)
        os.remove(path_j)

    return integrals
