"""Inverse Participation Ratio calculation.

Index
-----
.. currentmodule:: nanoqm.workflows.workflow_ipr
.. autosummary::
    workflow_ipr

"""
__all__ = ['workflow_ipr']

import logging

import numpy as np
from scipy.linalg import sqrtm

from qmflows.parsers.xyzParser import readXYZ

from ..common import (DictConfig, h2ev, number_spherical_functions_per_atom,
                      retrieve_hdf5_data)
from ..integrals.multipole_matrices import compute_matrix_multipole
from .initialization import initialize
from .tools import compute_single_point_eigenvalues_coefficients

# Starting logger
LOGGER = logging.getLogger(__name__)


def workflow_ipr(config: DictConfig) -> np.ndarray:
    """Compute the Inverse Participation Ratio main function."""
    # Dictionary containing the general information
    config.update(initialize(config))

    # Checking if hdf5 contains the required eigenvalues and coefficientsa
    compute_single_point_eigenvalues_coefficients(config)

    # Logger info
    LOGGER.info("Starting IPR calculation.")

    # Get eigenvalues and coefficients from hdf5
    node_path_coefficients = 'coefficients/point_0/'
    node_path_eigenvalues = 'eigenvalues/point_0'
    atomic_orbitals = retrieve_hdf5_data(config.path_hdf5, node_path_coefficients)
    energies = retrieve_hdf5_data(config.path_hdf5, node_path_eigenvalues)
    energies *= h2ev  # To get them from Hartree to eV

    # Converting the xyz-file to a mol-file
    mol = readXYZ(config.path_traj_xyz)

    # Computing the overlap-matrix S and its square root
    overlap = compute_matrix_multipole(mol, config, 'overlap')
    squared_overlap = sqrtm(overlap)

    # Converting the coeficients from AO-basis to MO-basis
    transformed_orbitals = np.dot(squared_overlap, atomic_orbitals)

    # Now we add up the rows of the c_MO that belong to the same atom
    sphericals = number_spherical_functions_per_atom(
        mol,
        'cp2k',
        config.cp2k_general_settings["basis"],
        config.path_hdf5)  # Array with number of spherical orbitals per atom

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
    result[:, 0] = energies
    result[:, 1] = 1.0 / ipr
    np.savetxt('IPR.txt', result)
    return result
