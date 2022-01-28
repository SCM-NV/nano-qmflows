"""Compute couplings and molecular orbitals.

Use the `Noodles schedule method <http://nlesc.github.io/noodles>` to generate
a dependency graph for running the jobs.

Index
-----
.. currentmodule:: nanoqm.schedule.scheduleCoupling
.. autosummary::
    calculate_overlap
    lazy_couplings
    write_hamiltonians

API
---
.. autofunction:: calculate_overlap
.. autofunction:: lazy_couplings
.. autofunction:: write_hamiltonians

"""

from __future__ import annotations

import logging
import os
from os.path import join
# Types hint
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from noodles import schedule
from scipy.optimize import linear_sum_assignment
from qmflows.parsers import parse_string_xyz

from ..common import (DictConfig, Matrix, MolXYZ, Tensor3D, Vector, hbar,
                      femtosec2au, h2ev, is_data_in_hdf5, retrieve_hdf5_data,
                      store_arrays_in_hdf5)
from ..integrals import (calculate_couplings_3points,
                         calculate_couplings_levine,
                         compute_overlaps_for_coupling, correct_phases)
from ..integrals.nonAdiabaticCoupling import (compute_range_orbitals,
                                              read_overlap_data)

if TYPE_CHECKING:
    import numpy.typing as npt

# Starting logger
logger = logging.getLogger(__name__)

__all__ = ["calculate_overlap", "lazy_couplings", "write_hamiltonians"]


@schedule
def lazy_couplings(
    config: DictConfig,
    paths_overlaps: List[str],
) -> Tuple[npt.NDArray[np.int_], List[str]]:
    """Compute the Nonadibatic coupling.

    The coupling is computed sing a either 3-point approximation. See:
    The Journal of Chemical Physics 137, 22A514 (2012); doi: 10.1063/1.4738960

    or a 2Point approximation using an smoothing function:
    J. Phys. Chem. Lett. 2014, 5, 2351−2356; doi: 10.1021/jz5009449

    Notice that the states can cross frequently due to unavoided crossing and
    such crossing must be track. see:
    J. Chem. Phys. 137, 014512 (2012); doi: 10.1063/1.4732536

    Parameters
    ----------
    config:
        Configuration of the current task
    paths_overlaps:
        path to the HDF5 node where the overlaps are stored.

    Returns
    -------
    Tuple
        with the swaps and couplings in each point.

    """
    if config.tracking:
        fixed_phase_overlaps, swaps = compute_the_fixed_phase_overlaps(
            paths_overlaps, config)
    else:
        # Do not track the crossings
        mtx_0 = retrieve_hdf5_data(config.path_hdf5, paths_overlaps[0])
        _, dim = mtx_0.shape
        overlaps = np.stack(
            retrieve_hdf5_data(config.path_hdf5, paths_overlaps))
        nOverlaps, nOrbitals, _ = overlaps.shape
        swaps = np.tile(np.arange(nOrbitals), (nOverlaps + 1, 1))
        mtx_phases = compute_phases(overlaps, nOverlaps, dim)
        fixed_phase_overlaps = correct_phases(overlaps, mtx_phases)

    # Write the overlaps in text format
    if config.write_overlaps:
        logger.debug("Writing down the overlaps in ascii format")
        write_overlaps_in_ascii(fixed_phase_overlaps)

    # Compute the couplings using either the levine method
    # or the 3Points approximation
    coupling_algorithms = {'levine': (calculate_couplings_levine, 1),
                           '3points': (calculate_couplings_3points, 2)}
    # Choose an algorithm to compute the couplings
    fun_coupling, step = coupling_algorithms[config["algorithm"]]
    config["fun_coupling"] = fun_coupling

    # Number of couplings to compute
    nCouplings = fixed_phase_overlaps.shape[0] - step + 1
    couplings = [calculate_couplings(config, i, fixed_phase_overlaps)
                 for i in range(nCouplings)]

    return swaps, couplings


def compute_the_fixed_phase_overlaps(
    paths_overlaps: List[str],
    config: DictConfig,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Fix the phase of the overlaps.

    First track the unavoided crossings between Molecular orbitals and
    finally correct the phase for the whole trajectory.
    """
    number_of_frames = len(paths_overlaps)
    # Pasth to the overlap matrices after the tracking
    # and phase correction
    roots = [join(config.orbitals_type, f'overlaps_{i}')
             for i in range(config.enumerate_from, number_of_frames + config.enumerate_from)]
    paths_corrected_overlaps = [join(r, 'mtx_sji_t0_corrected') for r in roots]
    # Paths inside the HDF5 to the array containing the tracking of the
    # unavoided crossings
    path_swaps = join(config.orbitals_type, 'swaps')

    # Compute the corrected overlaps if not avaialable in the HDF5
    all_data_in_hdf5 = all(is_data_in_hdf5(config.path_hdf5, path_data)
                           for path_data in (paths_corrected_overlaps[0], path_swaps))
    if not all_data_in_hdf5:
        # Compute the dimension of the coupling matrix
        mtx_0 = retrieve_hdf5_data(config.path_hdf5, paths_overlaps[0])
        _, dim = mtx_0.shape

        # Read all the Overlaps
        overlaps = np.stack([retrieve_hdf5_data(config.path_hdf5, ps)
                             for ps in paths_overlaps])

        # Number of couplings to compute
        nCouplings = overlaps.shape[0]

        # Compute the unavoided crossing using the Overlap matrix
        # and correct the swaps between Molecular Orbitals
        logger.debug("Computing the Unavoided crossings, "
                     "Tracking the crossings between MOs")
        overlaps, swaps = track_unavoided_crossings(overlaps, config.nHOMO)

        # Compute all the phases taking into account the unavoided crossings
        logger.debug("Computing the phases of the MOs")
        mtx_phases = compute_phases(overlaps, nCouplings, dim)

        # Fixed the phases of the whole set of overlap matrices
        fixed_phase_overlaps = correct_phases(overlaps, mtx_phases)

        # Store corrected overlaps in the HDF5
        store_arrays_in_hdf5(config.path_hdf5, paths_corrected_overlaps,
                             fixed_phase_overlaps)

        # Store the Swaps tracking the crossing
        store_arrays_in_hdf5(config.path_hdf5, path_swaps, swaps, dtype=np.int32)
    else:
        # Read the corrected overlaps and the swaps from the HDF5
        fixed_phase_overlaps = np.stack(
            retrieve_hdf5_data(config.path_hdf5, paths_corrected_overlaps))
        swaps = retrieve_hdf5_data(config.path_hdf5, path_swaps)

    return fixed_phase_overlaps, swaps


def calculate_couplings(config: DictConfig, i: int, fixed_phase_overlaps: Tensor3D) -> str:
    """Compute couplings for the i-th geometry.

    Search for the ith Coupling in the HDF5, if it is not available compute it
    using the 3 points approximation and store it in the HDF5.

    Returns
    -------
    str
        Path to the couplings store in the HDF5 file.

    """
    # time in atomic units
    dt_au = config.dt * femtosec2au

    # Path were the couplinp is store
    k = i + config.enumerate_from
    path = join(config.orbitals_type, f'coupling_{k}')

    # Skip the computation if the coupling is already done
    if is_data_in_hdf5(config.path_hdf5, path):
        logger.info(f"Coupling: {path} has already been calculated")
        return path
    else:
        logger.info(f"Computing coupling: {path}")
        if config.algorithm == 'levine':
            # Extract the overlap matrices involved in the coupling computation
            sji_t0 = fixed_phase_overlaps[i]
            # Compute the couplings with the phase corrected overlaps
            couplings = config["fun_coupling"](
                dt_au, sji_t0, sji_t0.transpose())
        elif config.algorithm == '3points':
            sji_t0, sji_t1 = fixed_phase_overlaps[i: i + 2]
            couplings = config["fun_coupling"](dt_au, sji_t0, sji_t0.transpose(), sji_t1,
                                               sji_t1.transpose())

            # Store the Coupling in the HDF5
        store_arrays_in_hdf5(config.path_hdf5, path, couplings)

        return path


def compute_phases(overlaps: Tensor3D, nCouplings: int,
                   dim: int) -> Matrix:
    """Compute the phase of the state_i at time t + dt.

    It uses the following equation:
    .. python::
        phase_i(t+dt) = Sii(t) * phase_i(t)
    """
    # initial references
    references = np.ones(dim)

    # Matrix containing the phases
    mtx_phases = np.empty((nCouplings + 1, dim))
    mtx_phases[0, :] = references

    # Compute the phases at times t + dt using the phases at time t
    for i in range(nCouplings):
        Sji_t = overlaps[i].reshape(dim, dim)

        # Compute the phase at time t
        phases = np.sign(np.diag(Sji_t)) * references
        mtx_phases[i + 1] = phases
        references = phases

    # Print phases (debug)
    logger.info(f"mtx_phases are stored in HDF5 node: {mtx_phases}")

    return mtx_phases


def track_unavoided_crossings(
    overlaps: npt.NDArray[np.float64],
    nHOMO: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Track the index of the states if there is a crossing.

    It uses the algorithm  described at:
    J. Chem. Phys. 137, 014512 (2012); doi: 10.1063/1.4732536.

    Returns
    -------
    tuple
        containing the corrected overlaps and the crossings

    """
    # 3D array containing the costs
    # Notice that the cost is compute on half of the overlap matrices
    # correspoding to Sji_t, the other half corresponds to Sij_t
    nOverlaps, nOrbitals, _ = overlaps.shape

    # Indexes taking into account the crossing
    # There are 2 Overlap matrices at each time t
    indexes = np.empty((nOverlaps + 1, nOrbitals), dtype=np.int_)
    indexes[0] = np.arange(nOrbitals, dtype=np.int_)

    # Track the crossing using the overlap matrices

    for k in range(nOverlaps):
        # Cost matrix to track the corssings
        logger.info(f"Tracking crossings at time: {k}")
        cost_mtx_homos = np.negative(overlaps[k, :nHOMO, :nHOMO] ** 2)
        cost_mtx_lumos = np.negative(overlaps[k, nHOMO:, nHOMO:] ** 2)

        # Compute the swap at time t + dt using two set of Orbitals:
        # HOMOs and LUMOS
        swaps_homos = linear_sum_assignment(cost_mtx_homos)[1]
        swaps_lumos = linear_sum_assignment(cost_mtx_lumos)[1]
        total_swaps = np.concatenate((swaps_homos, swaps_lumos + nHOMO))
        indexes[k + 1] = total_swaps

        # update the overlaps at times > t with the previous swaps
        if k != (nOverlaps - 1):  # last element
            k1 = k + 1
            # Update the matrix Sji at time t
            overlaps[k] = swap_columns(overlaps[k], total_swaps)
            # Update all the matrices Sji at time > t
            overlaps[k1:] = swap_forward(overlaps[k1:], total_swaps)
    # Accumulate the swaps
    acc = indexes[0]
    arr = np.empty(indexes.shape, dtype=np.int_)
    arr[0] = acc

    # Fold accumulating the crossings
    for i in range(nOverlaps):
        acc = acc[indexes[i + 1]]
        arr[i + 1] = acc

    return overlaps, arr


def swap_forward(
    overlaps: npt.NDArray[np.float64],
    swaps: npt.NDArray[np.integer],
) -> npt.NDArray[np.float64]:
    """Track all the crossings that happend previous to the current time.

    Swap the index i corresponding to the ith Molecular orbital
    with the corresponding swap at time t0.
    Repeat the same procedure with the index and swap at time t1.
    The swaps are compute with the linear sum assignment algorithm from Scipy.
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html

    """
    for i, mtx in enumerate(np.rollaxis(overlaps, 0)):
        overlaps[i] = mtx[:, swaps][swaps]

    return overlaps


@schedule
def calculate_overlap(config: DictConfig, mo_paths_hdf5: List[str]) -> List[str]:
    """Calculate the Overlap matrices.

    Parameters
    ----------
    config
        Configuration of the current job
    mo_paths_hdf5
        Node paths to the molecular orbitals in the HDF5

    Returns
    -------
        Node paths to the overlaps stored in the HDF5
    """
    # Number of couplings to compute
    npoints = len(config.geometries) - 1
    # Check what are the missing Couplings
    all_overlaps_paths = [create_overlap_path(
        config, i) for i in range(npoints)]
    overlap_is_done = [check_if_overlap_is_done(
        config, p) for p in all_overlaps_paths]

    paths = []
    for i in range(npoints):
        if overlap_is_done[i]:
            p = all_overlaps_paths[i]
        else:
            p = single_machine_overlaps(config, mo_paths_hdf5, i)
        paths.append(p)

    return paths


def single_machine_overlaps(
        config: DictConfig, mo_paths_hdf5: List[str], i: int) -> str:
    """Compute the overlaps in the CPUs avaialable on the local machine.

    Returns
    -------
    str
        Node path to the overlaps store in the HDF5

    """
    # Data to compute the overlaps
    pair_molecules = select_molecules(config, i)
    mo_paths = [mo_paths_hdf5[i + j][1] for j in range(2)]
    coefficients = read_overlap_data(config, mo_paths)

    # Compute the overlap
    overlaps = compute_overlaps_for_coupling(
        config, pair_molecules, coefficients)

    logger.info(f"overlap for point {i} was sucessfully computed!")

    # Store the array in the HDF5
    overlaps_paths_hdf5 = create_overlap_path(config, i)
    store_arrays_in_hdf5(config.path_hdf5, overlaps_paths_hdf5, overlaps)

    return overlaps_paths_hdf5


def create_overlap_path(config: DictConfig, i: int) -> str:
    """Create the path inside the HDF5 where the overlap is going to be store."""
    root = join(config.orbitals_type, 'overlaps_{}'.format(
        i + config.enumerate_from))
    return join(root, 'mtx_sji_t0')


def select_molecules(config: DictConfig, i: int) -> Tuple[MolXYZ, MolXYZ]:
    """Select the pairs of molecules to compute the couplings."""
    k = 0 if config.overlaps_deph else i
    return (
        parse_string_xyz(config.geometries[k]),
        parse_string_xyz(config.geometries[i + 1]),
    )


def check_if_overlap_is_done(config: DictConfig, overlaps_paths_hdf5: str) -> bool:
    """Search for a given Overlap inside the HDF5."""
    if is_data_in_hdf5(config.path_hdf5, overlaps_paths_hdf5):
        logger.info(f"{overlaps_paths_hdf5} Overlaps are already in the HDF5")
        return True
    else:
        logger.info(f"Computing: {overlaps_paths_hdf5}")
        return False


def write_hamiltonians(
        config: DictConfig,
        crossing_and_couplings: Tuple[np.ndarray, List[Matrix]],
        mo_paths_hdf5: List[str]) -> List[Tuple[str, str]]:
    """Write the real and imaginary components of the hamiltonian.

    It uses both the orbitals energies and the derivative coupling accoring to:
    http://pubs.acs.org/doi/abs/10.1021/ct400641n
    .. Note::
        **Units are: Rydbergs**.

    Parameters
    ----------
    config
        Configuration of the current task
    crossing_and-couplings
        Tuple of the states' crossings and couplings
    mo_path_hdf5
        Node path to the molecular orbitals in the HDF5

    Returns
    -------
    tuple
        Files containing the Hamiltonian Real and imaginary components

    """
    swaps, path_couplings = crossing_and_couplings
    nHOMO = config.nHOMO
    mo_index_range = config.mo_index_range

    def write_pyxaid_format(arr, fileName):
        np.savetxt(fileName, arr, fmt='%10.5e', delimiter='  ')

    def write_data(i):
        j = i + config.enumerate_from
        path_coupling = path_couplings[i]
        css = retrieve_hdf5_data(config.path_hdf5, path_coupling)

        # Extract the energy values at time t
        # The first coupling is compute at time t + dt
        # Then I'm shifting the energies dt to get the correct value
        energies_t0 = retrieve_hdf5_data(config.path_hdf5, mo_paths_hdf5[i][0])
        energies_t1 = retrieve_hdf5_data(
            config.path_hdf5, mo_paths_hdf5[i + 1][0])

        # Return the average between time t and t + dt
        energies = np.average((energies_t0, energies_t1), axis=0)

        # Print Energies in the range given by the user
        if all(x is not None for x in [nHOMO, mo_index_range]):
            lowest, highest = compute_range_orbitals(config)
            energies = energies[lowest: highest]

        # Swap the energies of the states that are crossing
        energies = energies[swaps[i + 1]]

        # FileNames
        path_dir_results = config.path_hamiltonians
        file_ham_im = join(path_dir_results, f'Ham_{i}_im')
        file_ham_re = join(path_dir_results, f'Ham_{j}_re')

        # Time units are atomic units. Convert them in fs-1, then in eV by hbar (eV * fs)
        ham_im = css * femtosec2au * hbar

        # Set the diagonal of the imaginary matrix to 0
        np.fill_diagonal(ham_im, 0)

        # Energies in eV
        ham_re = np.diag(h2ev * energies)

        write_pyxaid_format(ham_im, file_ham_im)
        write_pyxaid_format(ham_re, file_ham_re)

        return (file_ham_im, file_ham_re)

    # The couplings are compute at time t + dt therefore
    # we associate the energies at time t + dt with the corresponding coupling
    return [write_data(i) for i in range(config.npoints)]


def swap_columns(arr: Matrix, swaps_t: Vector) -> Matrix:
    """Swap only columns at t + dt to reconstruct the original overlap matrix."""
    return np.transpose(np.transpose(arr)[swaps_t])


def write_overlaps_in_ascii(overlaps: Tensor3D) -> None:
    """Write the corrected overlaps in text files."""
    if not os.path.isdir('overlaps'):
        os.mkdir('overlaps')

    # write overlaps
    nframes = overlaps.shape[0]
    for k in range(nframes):
        mtx_Sji = overlaps[k]
        path_Sji = f'overlaps/mtx_Sji_{k}'

        np.savetxt(path_Sji, mtx_Sji, fmt='%10.5e', delimiter='  ')
