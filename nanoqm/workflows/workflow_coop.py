"""Crystal Orbital Overlap Population calculation.

Index
-----
.. currentmodule:: nanoqm.workflows.workflow_coop
.. autosummary::
    workflow_crystal_orbital_overlap_population

"""

from __future__ import annotations

__all__ = ['workflow_crystal_orbital_overlap_population']

import logging
from typing import Tuple, TYPE_CHECKING

import numpy as np

from qmflows.parsers.xyzParser import readXYZ

from ..common import (DictConfig, MolXYZ, h2ev,
                      number_spherical_functions_per_atom, retrieve_hdf5_data)
from ..integrals.multipole_matrices import compute_matrix_multipole
from .initialization import initialize
from .tools import compute_single_point_eigenvalues_coefficients

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import float64 as f8, int64 as i8

# Starting logger
LOGGER = logging.getLogger(__name__)


def workflow_crystal_orbital_overlap_population(config: DictConfig) -> NDArray[f8]:
    """Compute the Crystal Orbital Overlap Population."""
    # Dictionary containing the general information
    config.update(initialize(config))

    # Checking hdf5 for eigenvalues and coefficients. If not present, they are
    # computed.
    compute_single_point_eigenvalues_coefficients(config)

    # Logger info
    LOGGER.info("Starting COOP calculation.")

    # Get eigenvalues and coefficients from hdf5
    atomic_orbitals, energies = get_eigenvalues_coefficients(config)

    # Converting the xyz-file to a mol-file
    mol = readXYZ(config.path_traj_xyz)

    # Computing the indices of the atomic orbitals of the two selected
    # elements, and the overlap matrix that contains only elements related to
    # the two elements
    el_1_orbital_ind, el_2_orbital_ind, overlap_reduced = compute_overlap_and_atomic_orbitals(
        mol, config)

    # Compute the crystal orbital overlap population between the two selected
    # elements
    coop = compute_coop(
        atomic_orbitals,
        overlap_reduced,
        el_1_orbital_ind,
        el_2_orbital_ind,
    )

    # Lastly, we save the COOP as a txt-file
    result_coop = print_coop(energies, coop)
    LOGGER.info("COOP calculation completed.")

    return result_coop


def get_eigenvalues_coefficients(config: DictConfig) -> Tuple[NDArray[f8], NDArray[f8]]:
    """Retrieve eigenvalues and coefficients from hdf5 file."""
    # Define paths to eigenvalues and coefficients hdf5
    node_path_coefficients = 'coefficients/point_0/'
    node_path_eigenvalues = 'eigenvalues/point_0'

    # Retrieves eigenvalues and coefficients
    atomic_orbitals = retrieve_hdf5_data(config.path_hdf5, node_path_coefficients)
    energies = retrieve_hdf5_data(config.path_hdf5, node_path_eigenvalues)

    # Energies converted from Hartree to eV
    energies *= h2ev

    # Return atomic orbitals and energies
    return atomic_orbitals, energies


def compute_overlap_and_atomic_orbitals(
    mol: MolXYZ,
    config: DictConfig,
) -> Tuple[NDArray[i8], NDArray[i8], NDArray[f8]]:
    """Compute the indices of the atomic orbitals of the two selected elements.

    Computes the overlap matrix, containing only the elements related to those two elements.
    """
    # Computing the overlap-matrix S
    overlap = compute_matrix_multipole(mol, config, 'overlap')

    # Computing number of spherical orbitals per atom
    sphericals = number_spherical_functions_per_atom(
        mol,
        'cp2k',
        config["cp2k_general_settings"]["basis"],
        config["path_hdf5"])

    # Getting the indices for the two selected elements
    element_1 = config["coop_elements"][0]
    element_2 = config["coop_elements"][1]

    element_1_index = [i for i, s in enumerate(mol) if element_1.lower() in s]
    element_2_index = [i for i, s in enumerate(mol) if element_2.lower() in s]

    # Making a list of the indices of the atomic orbitals for each of the two
    # elements
    atom_indices = np.zeros(len(mol) + 1, dtype=np.int64)
    atom_indices[1:] = np.cumsum(sphericals)

    _el_1_orbital_ind = [
        np.arange(sphericals[i], dtype=np.int64) + atom_indices[i] for i in element_1_index
    ]
    el_1_orbital_ind = np.reshape(
        _el_1_orbital_ind,
        len(element_1_index) * sphericals[element_1_index[0]],
    )

    _el_2_orbital_ind = [
        np.arange(sphericals[i], dtype=np.int64) + atom_indices[i] for i in element_2_index
    ]
    el_2_orbital_ind = np.reshape(
        _el_2_orbital_ind,
        len(element_2_index) * sphericals[element_2_index[0]],
    )

    # Reduced overlap matrix, containing only the elements related to the
    # overlap between element_1 and element_2
    # First select all the rows that belong to element_1
    overlap_reduced = overlap[el_1_orbital_ind, :]
    # Then select from those rows the columns that belong to species element_2
    overlap_reduced = overlap_reduced[:, el_2_orbital_ind]

    # Return lists of indices of atomic orbitals, and the reduced overlap
    # matrix
    return el_1_orbital_ind, el_2_orbital_ind, overlap_reduced


def compute_coop(
    atomic_orbitals: NDArray[f8],
    overlap_reduced: NDArray[f8],
    el_1_orbital_ind: NDArray[i8],
    el_2_orbital_ind: NDArray[i8],
) -> NDArray[f8]:
    """Define the function that computes the crystal orbital overlap population.

    Applying it to each column of the coefficent matrix.
    """
    # Define a function to be applied to each column of the coefficient matrix
    def coop_func(column_of_coefficient_matrix: NDArray[f8]) -> NDArray[f8]:
        # Multiply each coefficient-product with the relevant overlap, and sum
        # everything
        return np.sum(
            np.tensordot(
                column_of_coefficient_matrix[el_1_orbital_ind],
                column_of_coefficient_matrix[el_2_orbital_ind],
                0) * overlap_reduced)

    # Call the function
    coop = np.apply_along_axis(
        coop_func,
        0,
        atomic_orbitals)

    # Return the calculated crystal orbital overlap population
    return coop


def print_coop(energies: NDArray[f8], coop: NDArray[f8]) -> NDArray[f8]:
    """Save the COOP in a txt-file."""
    result_coop = np.zeros((len(coop), 2))
    result_coop[:, 0], result_coop[:, 1] = energies, coop
    np.savetxt('COOP.txt', result_coop)

    return result_coop
