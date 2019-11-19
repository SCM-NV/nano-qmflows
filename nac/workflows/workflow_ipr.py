"""Inverse Participation Ratio calculation."""
__all__ = ['workflow_ipr']

import logging
import numpy as np
from qmflows.parsers.xyzParser import readXYZ
from scipy.constants import physical_constants
from scipy.linalg import sqrtm

from nac.workflows import workflow_single_points
from nac.common import (
    number_spherical_functions_per_atom,
    retrieve_hdf5_data, is_data_in_hdf5)
from nac.integrals.multipole_matrices import compute_matrix_multipole
from nac.workflows.initialization import initialize

# Starting logger
LOGGER = logging.getLogger(__name__)


def workflow_ipr(config: dict) -> list:
    """Inverse Participation Ratio main function."""
    # Dictionary containing the general information
    config.update(initialize(config))

    # Checking if hdf5 contains the required eigenvalues and coefficients
    path_coefficients = '{}/point_0/cp2k/mo/coefficients'.format(
        config["project_name"])
    path_eigenvalues = '{}/point_0/cp2k/mo/eigenvalues'.format(
        config["project_name"])

    predicate_1 = is_data_in_hdf5(config["path_hdf5"], path_coefficients)
    predicate_2 = is_data_in_hdf5(config["path_hdf5"], path_eigenvalues)
    if all((predicate_1, predicate_2)):
        LOGGER.info("Coefficients and eigenvalues already in hdf5.")
    else:
        # Call the single point workflow to calculate the eigenvalues and
        # coefficients
        LOGGER.info("Starting single point calculation.")
        workflow_single_points(config)

    # Logger info
    LOGGER.info("Starting IPR calculation.")

    # Get eigenvalues and coefficients from hdf5
    atomic_orbitals = retrieve_hdf5_data(config["path_hdf5"], path_coefficients)
    energies = retrieve_hdf5_data(config["path_hdf5"], path_eigenvalues)

    h2ev = physical_constants['Hartree energy in eV'][0]
    energies = energies * h2ev  # To get them from Hartree to eV

    # Converting the xyz-file to a mol-file
    mol = readXYZ(config["path_traj_xyz"])

    # Computing the overlap-matrix S and its square root
    overlap = compute_matrix_multipole(mol, config, 'overlap')
    squared_overlap = sqrtm(overlap)

    # Converting the coeficients from AO-basis to MO-basis
    transformed_orbitals = np.dot(squared_overlap, atomic_orbitals)

    # Now we add up the rows of the c_MO that belong to the same atom
    sphericals = number_spherical_functions_per_atom(
        mol,
        'cp2k',
        config["cp2k_general_settings"]["basis"],
        config["path_hdf5"])  # Array with number of spherical orbitals per atom

    # New matrix with the atoms on the rows and the MOs on the columns
    indices = np.zeros(len(mol), dtype='int')
    indices[1:] = np.cumsum(sphericals[:-1])
    accumulated_transf_orbitals = np.add.reduceat(transformed_orbitals, indices, 0)

    # Finally, we can calculate the IPR
    ipr = np.zeros(accumulated_transf_orbitals.shape[1])

    for i in range(accumulated_transf_orbitals.shape[1]):
        ipr[i] = np.sum(np.absolute(accumulated_transf_orbitals[:, i])**4) / \
            (np.sum(np.absolute(accumulated_transf_orbitals[:, i])**2))**2

    # Lastly, we save the output as a txt-file
    result = np.zeros((accumulated_transf_orbitals.shape[1], 2))
    result[:, 0], result[:, 1] = energies, 1 / ipr

    np.savetxt('IPR.txt', result)
    return result
