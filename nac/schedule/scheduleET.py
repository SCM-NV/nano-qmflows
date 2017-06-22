
from itertools import starmap
from nac.common import (
    Array, Matrix, Tensor3D, Vector, retrieve_hdf5_data, search_data_in_hdf5,
    store_arrays_in_hdf5)
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from nac.integrals.nonAdiabaticCoupling import calculate_spherical_overlap
from nac.integrals.spherical_Cartesian_cgf import calc_transf_matrix
from os.path import  join
from scipy import sparse
from typing import (Dict, List, Tuple)
import h5py
import numpy as np


def photo_excitation_rate(
        geometries: Tuple, tensor_overlaps: Tensor3D,
        time_dependent_coeffs: Matrix, map_index_pyxaid_hdf5: Matrix,
        dt_au: float) -> Tuple:
    """
    Calculate the Electron transfer rate, using both adiabatic and nonadiabatic
    components, using equation number 8 from:
    J. AM. CHEM. SOC. 2005, 127, 7941-7951. Ab Initio Nonadiabatic Molecular
    Dynamics of the Ultrafast Electron Injection acrossthe Alizarin-TiO2
    Interface.
    The derivatives are calculated numerically using 3 points.

    :param geometry: Molecular geometries.
    :type geometry: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :param tensor_overlaps: Overlap matrices at time t - dt, t and t + dt.
    :param time_dependent_coeffs: Time-dependentent coefficients
    at time t - dt, t and t + dt.
    :param map_index_pyxaid_hdf5: Index relation between the Excitations
    in PYXAID and the corresponding molecular orbitals store in the HDF5.
    :param dt_au: Delta time integration used in the dynamics.
    :returns: tuple containing both nonadiabatic and adiabatic components
    """
    # indices of the i -> j transitions used by PYXAID
    row_indices = map_index_pyxaid_hdf5[:, 0]
    col_indices = map_index_pyxaid_hdf5[:, 1]

    # Rearrange the overlap matrix in the PYXAID order
    matrix_overlap_pyxaid_order = tensor_overlaps[1][row_indices, col_indices]

    # NonAdiabatic component
    coeff_derivatives = np.apply_along_axis(
        lambda v: (v[0] - v[2]) / (2 * dt_au), 0, time_dependent_coeffs)

    nonadiabatic = np.sum(coeff_derivatives * matrix_overlap_pyxaid_order)

    # Adiabatic component
    overlap_derv = np.apply_along_axis(
        lambda v: (v[0] - v[2]) / (2 * dt_au), 0, tensor_overlaps)

    overlap_derv_pyxaid_order = overlap_derv[row_indices, col_indices]

    adiabatic = np.sum(time_dependent_coeffs[1] * overlap_derv_pyxaid_order)

    return nonadiabatic, adiabatic


def compute_overlaps_ET(
        project_name: str, molecules: List, basis_name: str,
        path_hdf5: str, dictCGFs: Dict, mo_paths_hdf5: List,
        fragment_indices: Array, enumerate_from: int,
        package_name: int) -> List:
    """
    Given a list of molecular fragments compute the Overlaps in the
    molecular orbital basis for the different fragments and for all the
    molecular frames.
    """
    # Convert the indices to a numpy array
    if not isinstance(fragment_indices, np.ndarray):
        fragment_indices = np.array(fragment_indices, dtype=np.int32)

    # Matrix containing the lower and upper range for the CGFs of each atom
    # in order.
    indices_range_CGFs = create_indices_range_CGFs(molecules[0], dictCGFs)

    fragment_overlaps = []
    for vector_indices in np.rollaxis(fragment_indices):
        # extract the atoms of the fragment
        frames_fragment_atoms = [[mol[i] for i in vector_indices]
                                 for mol in molecules]
        # compute the transformation matrix from cartesian to spherical
        sparse_trans_mtx = compute_fragment_trans_mtx(
            path_hdf5, basis_name, frames_fragment_atoms[0], package_name)

        # create a Hash for the fragment
        fragment_hash = hash(vector_indices.tostring())

        # Compute the indices of the MOs corresponding to the atoms in the
        # fragments
        indices_fragment_mos = compute_indices_fragments_mos(
            vector_indices, indices_range_CGFs)

        # Overlap matrix for the given fragment
        overlaps = compute_frames_fragment_overlap(
            project_name, frames_fragment_atoms, path_hdf5, mo_paths_hdf5,
            indices_fragment_mos, dictCGFs, sparse_trans_mtx, enumerate_from,
            fragment_hash)
        fragment_overlaps.append(overlaps)

    return fragment_overlaps


def compute_frames_fragment_overlap(
        project_name: str, molecules: str, path_hdf5: str, mo_paths_hdf5: List,
        indices_fragment_mos: Vector, dictCGFs: Dict, sparse_trans_mtx: Matrix,
        enumerate_from: int, fragment_hash: int) -> List:

    """
    Compute all the overlap matrices for the frames of a molecular dynamics
    for a given fragment.
    """
    # Path to store the fragment overlap matrices in the HDF5
    range_points = range(enumerate_from, enumerate_from + len(molecules))
    paths = [join(project_name, 'point_{}'.format(i),
                  'fragment_overlap_hash_{}'.format(fragment_hash))
             for i in range_points]

    return [compute_fragment_overlap(
        path_hdf5, p, mol, path_mos, indices_fragment_mos, dictCGFs,
        sparse_trans_mtx)
        for p, mol, path_mos in zip(paths, molecules, mo_paths_hdf5)]


def compute_fragment_overlap(
        path_hdf5: str, path_overlap: str, mol: List, path_mos: str,
        indices_fragment_mos: Vector, dictCGFs: Dict, fragment_indices: Vector,
        trans_mtx: Matrix) -> str:
    """
    Compute the overlap matrix only for those atoms included in the fragment
    """
    if not search_data_in_hdf5(path_hdf5, path_overlap):
        # Extract atoms belonging to the fragment
        fragment_atoms = [mol[i] for i in fragment_indices]
        # Compute the overlap in the atomic basis
        overlap_AO = calcMtxMultipoleP(fragment_atoms, dictCGFs)

        # Read all the molecular orbital coefficients
        coefficients = retrieve_hdf5_data(path_hdf5, path_mos[1])
        # Extract the coefficients belonging to the fragment
        dim = indices_fragment_mos.size
        fragment_coefficients = coefficients[
            np.repeat(indices_fragment_mos, dim),
            np.tile(indices_fragment_mos, dim)].reshape(dim, dim)

        # Compute the overlap in spherical coordinates
        overlap_spherical = calculate_spherical_overlap(
            trans_mtx, overlap_AO, fragment_coefficients, fragment_coefficients)
        store_arrays_in_hdf5(path_hdf5, path_overlap, overlap_spherical)

    return path_overlap


def create_indices_range_CGFs(molecule: List, dictCGFs: Dict) -> Matrix:
    """
    Creates a matrix containing the lower and upper(exclusive) indices
    of the CGFs for each atoms in order.
    """
    ranges = np.empty((len(molecule), 2), dtype=np.int32)

    lower = 0
    for i, at in enumerate(molecule):
        upper = lower + len(dictCGFs[at.symbol])
        ranges[i] = lower, upper
        lower = upper

    return ranges


def compute_fragment_trans_mtx(
        path_hdf5: str, basis_name: str, fragment_atoms: List,
        package_name: str) -> Matrix:
    """
    Calculate the cartesian to spherical transformation matrix for
    the given fragment
    """
    with h5py.File(path_hdf5, 'r') as f5:
        trans_mtx = calc_transf_matrix(
            f5, fragment_atoms, basis_name, package_name)

    return sparse.csr_matrix(trans_mtx)


def compute_indices_fragments_mos(
        vector_indices: Vector, indices_range_CGFs: Matrix) -> Vector:
    """
    Compute the indices of the CGFs of the fragments to extract from
    the coefficients stored in the HDF5.
    """
    # extract the ranges of CGFs for the fragments
    fragment_ranges = indices_range_CGFs[vector_indices]

    return np.concatenate(list(starmap(np.arange, fragment_ranges)))
