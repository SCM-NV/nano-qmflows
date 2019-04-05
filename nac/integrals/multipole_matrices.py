from compute_integrals import compute_integrals_multipole
from nac.common import (
    Matrix, retrieve_hdf5_data, is_data_in_hdf5,
    store_arrays_in_hdf5, tuplesXYZ_to_plams)
from os.path import join
import logging
import os
import uuid

# Starting logger
logger = logging.getLogger(__name__)


def get_multipole_matrix(config: dict, inp: dict, multipole: str) -> Matrix:
    """
    Retrieve the `multipole` number `i` from the trajectory. Otherwise compute it.
    """
    root = join(config['project_name'], 'multipole', 'point_{}'.format(inp.i))
    path_hdf5 = config['path_hdf5']
    path_multipole_hdf5 = join(root, multipole)
    matrix_multipole = search_multipole_in_hdf5(path_hdf5, path_multipole_hdf5, multipole)

    if matrix_multipole is None:
        matrix_multipole = compute_matrix_multipole(inp.mol, config, multipole)
        store_arrays_in_hdf5(path_hdf5, path_multipole_hdf5, matrix_multipole)

    return matrix_multipole


def search_multipole_in_hdf5(path_hdf5: str, path_multipole_hdf5: str, multipole: str):
    """
    Search if the multipole is already store in the HDFt
    """
    if is_data_in_hdf5(path_hdf5, path_multipole_hdf5):
        logger.info("retrieving multipole: {} from the hdf5".format(multipole))
        return retrieve_hdf5_data(path_hdf5, path_multipole_hdf5)
    else:
        logger.info("computing multipole: {}".format(multipole))
        return None


def compute_matrix_multipole(
        mol: list, config: dict, multipole: str) -> Matrix:
    """
    Compute the some `multipole` matrix: overlap, dipole, etc. for a given geometry `mol`.
    Compute the Multipole matrix in spherical coordinates.

    Note: for the dipole onwards the super_matrix contains all the matrices stack all the
    0-axis.

    :returns: Matrix with entries <ψi | x^i y^j z^k | ψj>
    """
    path_hdf5 = config['path_hdf5']

    # Write molecule in temporal file
    path = join(config["scratch_path"], "molecule_{}.xyz".format(uuid.uuid4()))
    mol_plams = tuplesXYZ_to_plams(mol)
    mol_plams.write(path)

    # name of the basis set
    basis_name = config["cp2k_general_settings"]["basis"]

    if multipole == 'overlap':
        matrix_multipole = compute_integrals_multipole(path, path_hdf5, basis_name, multipole)
    elif multipole == 'dipole':
        # The tensor contains the overlap + {x, y, z} dipole matrices
        super_matrix = compute_integrals_multipole(path, path_hdf5, basis_name, multipole)
        dim = super_matrix.shape[1]

        # Reshape the super_matrix as a tensor containing overlap + {x, y, z} dipole matrices
        matrix_multipole = super_matrix.reshape(4, dim, dim)

    elif multipole == 'quadrupole':
        # The tensor contains the overlap + {xx, xy, xz, yy, yz, zz} quadrupole matrices
        super_matrix = compute_integrals_multipole(path, path_hdf5, basis_name, multipole)
        dim = super_matrix.shape[1]

        # Reshape to 3d tensor containing overlap + {xx, xy, xz, yy, yz, zz} quadrupole matrices
        matrix_multipole = super_matrix.reshape(7, dim, dim)

    # Delete the tmp molecule file
    os.remove(path)

    return matrix_multipole
