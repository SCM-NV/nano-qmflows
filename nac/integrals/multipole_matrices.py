from nac.common import (
    Matrix, compute_center_of_mass, retrieve_hdf5_data, search_data_in_hdf5,
    store_arrays_in_hdf5, triang2mtx)
from nac.integrals.overlapIntegral import calcMtxOverlapP
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from os.path import join
from scipy import sparse
from typing import (Dict, List)
import numpy as np


def get_multipole_matrix(mol: List, config: Dict, multipole: str) -> Matrix:
    """
    """
    root = join(config['project_name'], 'multipole')
    path_hdf5 = config['path_hdf5']
    path_multipole_hdf5 = join(root, multipole)
    matrix_multipole = search_multipole_in_hdf5(path_hdf5, path_multipole_hdf5, multipole)

    if matrix_multipole is None:
        matrix_multipole = compute_matrix_multipole(mol, config, multipole)

    store_arrays_in_hdf5(path_hdf5, path_multipole_hdf5, matrix_multipole)

    return matrix_multipole


def search_multipole_in_hdf5(path_hdf5: str, path_multipole_hdf5: str, multipole: str):
    """
    Search if the multipole is already store in the HDFt
    """
    if search_data_in_hdf5(path_hdf5, path_multipole_hdf5):
        print("retrieving multipole: {} from the hdf5".format(multipole))
        return retrieve_hdf5_data(path_hdf5, path_multipole_hdf5)
    else:
        print("computing multipole: {}".format(multipole))
        return None


def compute_matrix_multipole(
        mol: List, config: Dict, multipole: str) -> Matrix:
    """
    Compute the some `multipole` matrix: overlap, dipole, etc. for a given geometry `mol`.
    """
    path_hdf5 = config['path_hdf5']
    runner = config['runner']

    # Compute the number of cartesian basis functions
    dictCGFs = config['dictCGFs']
    n_cart_funcs = np.sum(np.stack(len(dictCGFs[at.symbol]) for at in mol))

    # Compute the transformation matrix from cartesian to spherical
    transf_mtx = retrieve_hdf5_data(path_hdf5, config['hdf5_trans_mtx'])
    transf_mtx = sparse.csr_matrix(transf_mtx)
    transpose = transf_mtx.transpose()

    if multipole == 'overlap':
        rs = calcMtxOverlapP(mol, dictCGFs, runner)
        mtx_overlap = triang2mtx(rs, n_cart_funcs)  # there are 1452 Cartesian basis CGFs
        matrix_multipole = transf_mtx.dot(sparse.csr_matrix.dot(mtx_overlap, transpose))

    else:
        rc = compute_center_of_mass(mol)
        exponents = {
            'dipole': [
                {'e': 1, 'f': 0, 'g': 0}, {'e': 0, 'f': 1, 'g': 0}, {'e': 0, 'f': 0, 'g': 1}],
            'quadrupole': [
                {'e': 2, 'f': 0, 'g': 0}, {'e': 0, 'f': 2, 'g': 0}, {'e': 0, 'f': 0, 'g': 2}]
        }
        mtx_integrals_triang = tuple(calcMtxMultipoleP(mol, dictCGFs, runner, rc, **kw)
                                     for kw in exponents[multipole])
        mtx_integrals_cart = tuple(triang2mtx(xs, n_cart_funcs)
                                   for xs in mtx_integrals_triang)
        matrix_multipole = np.stack(
            transf_mtx.dot(sparse.csr_matrix.dot(x, transpose)) for x in mtx_integrals_cart)

    return matrix_multipole
