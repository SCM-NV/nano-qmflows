__all__ = ['workflow_ipr']

# Module for single_point workflow
from nac.workflows import workflow_single_points

# Modules for IPR caluclation
from nac.common import (
    number_spherical_functions_per_atom,
    retrieve_hdf5_data, is_data_in_hdf5)
from nac.integrals.multipole_matrices import compute_matrix_multipole
from nac.workflows.initialization import initialize
import numpy as np
from qmflows.parsers.xyzParser import readXYZ
from scipy.constants import physical_constants
from scipy.linalg import sqrtm
import logging

# Starting logger
logger = logging.getLogger(__name__)

def workflow_ipr(config: dict) -> list:

    # Dictionary containing the general information
    config.update(initialize(config))

    # Checking if hdf5 contains the required eigenvalues and coefficients
    path_coefficients = '{}/point_0/cp2k/mo/coefficients'.format(
        config["project_name"])
    path_eigenvalues = '{}/point_0/cp2k/mo/eigenvalues'.format(
        config["project_name"])

    if is_data_in_hdf5(
            config["path_hdf5"],
            path_coefficients) and is_data_in_hdf5(
            config["path_hdf5"],
            path_eigenvalues):
        logger.info("Coefficients and eigenvalues already in hdf5.")
    else:
        # Call the single point workflow to calculate the eigenvalues and
        # coefficients
        logger.info("Starting single point calculation.")
        workflow_single_points(config)

    # Logger info
    logger.info("Starting IPR calculation.")

    # Get eigenvalues and coefficients from hdf5
    c_AO = retrieve_hdf5_data(config["path_hdf5"], path_coefficients)
    Energies = retrieve_hdf5_data(config["path_hdf5"], path_eigenvalues)

    h2ev = physical_constants['Hartree energy in eV'][0]
    Energies = Energies * h2ev  # To get them from Hartree to eV

    # Converting the xyz-file to a mol-file
    mol = readXYZ(config["path_traj_xyz"])

    # Computing the overlap-matrix S and its square root
    S = compute_matrix_multipole(mol, config, 'overlap')
    Sm = sqrtm(S)

    # Converting the coeficients from AO-basis to MO-basis
    c_MO = np.dot(Sm, c_AO)

    # Now we add up the rows of the c_MO that belong to the same atom
    sphericals = number_spherical_functions_per_atom(
        mol,
        'cp2k',
        config["cp2k_general_settings"]["basis"],
        config["path_hdf5"])  # Array with number of spherical orbitals per atom

    # New matrix with the atoms on the rows and the MOs on the columns
    Indices = np.zeros(len(mol), dtype='int')
    Indices[1:] = np.cumsum(sphericals[:-1])
    c_MO_cont = np.add.reduceat(c_MO, Indices, 0)

    # Finally, we can calculate the IPR
    ipr = np.zeros(c_MO_cont.shape[1])

    for i in range(c_MO_cont.shape[1]):
        ipr[i] = np.sum(np.absolute(c_MO_cont[:, i])**4) / \
            (np.sum(np.absolute(c_MO_cont[:, i])**2))**2

    # Lastly, we save the output as a txt-file
    result = np.zeros((c_MO_cont.shape[1], 2))
    result[:, 0], result[:, 1] = Energies, 1 / ipr

    np.savetxt('IPR.txt', result)
    return result
