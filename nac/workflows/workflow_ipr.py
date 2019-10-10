__all__ = ['workflow_ipr']

# Module for single_point workflow
from nac.workflows import workflow_single_points

# Modules for IPR caluclation
import h5py
from nac.common import (
    number_spherical_functions_per_atom,
    retrieve_hdf5_data)
from nac.integrals.multipole_matrices import compute_matrix_multipole
import numpy as np
from qmflows.parsers.xyzParser import readXYZ
from scipy.linalg import sqrtm
from nac.workflows.initialization import initialize

import logging

# Starting logger
logger = logging.getLogger(__name__)


def workflow_ipr(config: dict) -> list:

    # Dictionary containt the general information
    config.update(initialize(config))

    # Logger info
    logger.info("Starting single point calculation.")

    # Call the single point workflow to calculate the eigenvalues and
    # coefficients
    hdf5_path = workflow_single_points(config)

    # Logger info
    logger.info("Starting IPR calculation.")

    # Get eigenvalues and coefficients from hdf5 file
    c_AO = retrieve_hdf5_data(config["path_hdf5"], hdf5_path[0][0][1])
    Energies = retrieve_hdf5_data(config["path_hdf5"], hdf5_path[0][0][0])
    Energies = Energies * 27.211  # To get them from Hartree to eV

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

    j = 0
    # New matrix with the atoms on the rows and the MOs on the columns
    c_MO_cont = np.zeros((len(sphericals), c_MO.shape[1]))

    for i in range(len(sphericals)):
        number = sphericals[i]
        # We add up all the rows that belong to the same atom
        c_MO_cont[i, :] = np.sum(c_MO[j:j + number, :], 0)
        j += number

    # Finally, we can calculate the IPR
    IPR = np.zeros(c_MO_cont.shape[1])

    for i in range(c_MO_cont.shape[1]):
        IPR[i] = np.sum(np.absolute(c_MO_cont[:, i])**4) / \
            (np.sum(np.absolute(c_MO_cont[:, i])**2))**2

    # Lastly, we save the output as a txt-file
    RESULT = np.zeros((c_MO_cont.shape[1], 2))
    RESULT[:, 0], RESULT[:, 1] = Energies, 1 / IPR

    np.savetxt('IPR.txt', RESULT)
    return RESULT
