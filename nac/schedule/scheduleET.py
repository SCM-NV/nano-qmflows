
from nac.common import (Matrix, Vector, retrieve_hdf5_data, search_data_in_hdf5,
                        store_arrays_in_hdf5)
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from nac.integrals.nonAdiabaticCoupling import calculate_spherical_overlap
from os.path import  join
from scipy import sparse
from typing import (Dict, List)


def compute_overlaps_ET(
        project_name: str, molecules: List, path_hdf5: str,
        mo_paths_hdf5: List, hdf5_trans_mtx: str, fragment_indices: Vector,
        dictCGFs: Dict, enumerate_from) -> List:
    """
    Given a list of molecular geometries and some indices for the fragment
    """
    # Read Cartesian to spherical transformation matrix
    trans_mtx = sparse.csr_matrix(
        retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx))

    # Path to store the fragment overlap matrices in the HDF5
    range_points = range(enumerate_from, enumerate_from + len(molecules))
    paths = [join(project_name, 'point_{}'.format(i), 'fragment_overlap')
             for i in range_points]

    return [compute_fragment_overlap(
        path_hdf5, p, mol, path_mos, dictCGFs, fragment_indices, trans_mtx)
        for p, mol, path_mos in zip(paths, molecules, mo_paths_hdf5)]


def compute_fragment_overlap(
        path_hdf5: str, path_overlap: str, mol: List, path_mos: str,
        dictCGFs: Dict, fragment_indices: Vector, trans_mtx: Matrix) -> str:
    """
    Compute the overlap matrix only for those atoms included in the fragment
    """
    if not search_data_in_hdf5(path_hdf5, path_overlap):
        # Extract atoms belonging to the fragment
        fragment_atoms = [mol[i] for i in fragment_indices]
        # Compute the overlap in the atomic basis
        overlap_AO = calcMtxMultipoleP(fragment_atoms, dictCGFs)
        # Read the molecular orbital coefficients
        coefficients = retrieve_hdf5_data(path_hdf5, path_mos[1])
        # Compute the overlap in spherical coordinates
        overlap_spherical = calculate_spherical_overlap(
            trans_mtx, overlap_AO, coefficients, coefficients)
        store_arrays_in_hdf5(path_hdf5, path_overlap, overlap_spherical)

    return path_overlap
