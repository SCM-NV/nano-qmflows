__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========

from os.path import join
from scipy.optimize import linear_sum_assignment

import logging
import numpy as np
import os

from nac.integrals import (
    calculate_couplings_levine, calculate_couplings_3points,
    compute_overlaps_for_coupling, correct_phases)
from nac.common import (
    Matrix, Vector, Tensor3D,
    femtosec2au, retrieve_hdf5_data,
    is_data_in_hdf5, store_arrays_in_hdf5)
from qmflows.parsers import parse_string_xyz

# Types hint
from typing import (List, Tuple)

# Starting logger
logger = logging.getLogger(__name__)

# ==============================> Schedule Tasks <=============================


def lazy_couplings(config: dict, paths_overlaps: list) -> list:
    """
    Compute the Nonadibatic coupling using a 3 point approximation. See:
    The Journal of Chemical Physics 137, 22A514 (2012); doi: 10.1063/1.4738960

    or a 2Point approximation using an smoothing function:
    J. Phys. Chem. Lett. 2014, 5, 2351−2356; doi: 10.1021/jz5009449

    Notice that the states can cross frequently due to unavoided crossing and
    such crossing must be track. see:
    J. Chem. Phys. 137, 014512 (2012); doi: 10.1063/1.4732536
    """
    if config.tracking:
        fixed_phase_overlaps, swaps = compute_the_fixed_phase_overlaps(
            paths_overlaps, config.path_hdf5, config.project_name,
            config.enumerate_from, config.nHOMO)
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
        paths_overlaps: List, path_hdf5: str, project_name: str,
        enumerate_from: int, nHOMO: int) -> Tuple:
    """
    First track the unavoided crossings between Molecular orbitals and
    finally correct the phase for the whole trajectory.
    """
    number_of_frames = len(paths_overlaps)
    # Pasth to the overlap matrices after the tracking
    # and phase correction
    roots = [join(project_name, 'overlaps_{}'.format(i))
             for i in range(enumerate_from, number_of_frames + enumerate_from)]
    paths_corrected_overlaps = [join(r, 'mtx_sji_t0_corrected') for r in roots]
    # Paths inside the HDF5 to the array containing the tracking of the
    # unavoided crossings
    path_swaps = join(project_name, 'swaps')

    # Compute the corrected overlaps if not avaialable in the HDF5
    if not is_data_in_hdf5(path_hdf5, paths_corrected_overlaps[0]):

        # Compute the dimension of the coupling matrix
        mtx_0 = retrieve_hdf5_data(path_hdf5, paths_overlaps[0])
        _, dim = mtx_0.shape

        # Read all the Overlaps
        overlaps = np.stack([retrieve_hdf5_data(path_hdf5, ps)
                             for ps in paths_overlaps])

        # Number of couplings to compute
        nCouplings = overlaps.shape[0]

        # Compute the unavoided crossing using the Overlap matrix
        # and correct the swaps between Molecular Orbitals
        logger.debug("Computing the Unavoided crossings, "
                     "Tracking the crossings between MOs")
        overlaps, swaps = track_unavoided_crossings(overlaps, nHOMO)

        # Compute all the phases taking into account the unavoided crossings
        logger.debug("Computing the phases of the MOs")
        mtx_phases = compute_phases(overlaps, nCouplings, dim)

        # Fixed the phases of the whole set of overlap matrices
        fixed_phase_overlaps = correct_phases(overlaps, mtx_phases)

        # Store corrected overlaps in the HDF5
        store_arrays_in_hdf5(path_hdf5, paths_corrected_overlaps,
                             fixed_phase_overlaps)

        # Store the Swaps tracking the crossing
        store_arrays_in_hdf5(path_hdf5, path_swaps, swaps, dtype=np.int32)
    else:
        # Read the corrected overlaps and the swaps from the HDF5
        fixed_phase_overlaps = np.stack(
            retrieve_hdf5_data(path_hdf5, paths_corrected_overlaps))
        swaps = retrieve_hdf5_data(path_hdf5, path_swaps)

    return fixed_phase_overlaps, swaps


def calculate_couplings(config: dict, i: int, fixed_phase_overlaps: Tensor3D) -> str:
    """
    Search for the ith Coupling in the HDF5, if it is not available compute it
    using the 3 points approximation and store it in the HDF5
    """
    # time in atomic units
    dt_au = config.dt * femtosec2au

    # Path were the couplinp is store
    k = i + config.enumerate_from
    path = join(config.project_name, 'coupling_{}'.format(k))

    # Skip the computation if the coupling is already done
    if is_data_in_hdf5(config.path_hdf5, path):
        logger.info("Coupling: {} has already been calculated".format(path))
        return path
    else:
        logger.info("Computing coupling: {}".format(path))
        if config.algorithm == 'levine':
            # Extract the overlap matrices involved in the coupling computation
            sji_t0 = fixed_phase_overlaps[i]
            # Compute the couplings with the phase corrected overlaps
            couplings = config["fun_coupling"](dt_au, sji_t0, sji_t0.transpose())
        elif config.algorithm == '3points':
            sji_t0, sji_t1 = fixed_phase_overlaps[i: i + 2]
            couplings = config["fun_coupling"](dt_au, sji_t0, sji_t0.transpose(), sji_t1,
                                               sji_t1.transpose())

            # Store the Coupling in the HDF5
        store_arrays_in_hdf5(config.path_hdf5, path, couplings)

        return path


def compute_phases(overlaps: Tensor3D, nCouplings: int,
                   dim: int) -> Matrix:
    """
    Compute the phase of the state_i at time t + dt, using the following
    equation:
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
    np.savetxt('mtx_phases', mtx_phases)

    return mtx_phases


def track_unavoided_crossings(overlaps: Tensor3D, nHOMO: int) -> Tuple:
    """
    Track the index of the states if there is a crossing using the
    algorithm  described at:
    J. Chem. Phys. 137, 014512 (2012); doi: 10.1063/1.4732536.
    """
    # 3D array containing the costs
    # Notice that the cost is compute on half of the overlap matrices
    # correspoding to Sji_t, the other half corresponds to Sij_t
    nOverlaps, nOrbitals, _ = overlaps.shape

    # Indexes taking into account the crossing
    # There are 2 Overlap matrices at each time t
    indexes = np.empty((nOverlaps + 1, nOrbitals), dtype=np.int)
    indexes[0] = np.arange(nOrbitals, dtype=np.int)

    # Track the crossing using the overlap matrices

    for k in range(nOverlaps):
        # Cost matrix to track the corssings
        logger.info("Tracking crossings at time: {}".format(k))
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
    arr = np.empty(indexes.shape, dtype=np.int)
    arr[0] = acc

    # Fold accumulating the crossings
    for i in range(nOverlaps):
        acc = acc[indexes[i + 1]]
        arr[i + 1] = acc

    return overlaps, arr


def swap_forward(overlaps: Tensor3D, swaps: Vector) -> Tensor3D:
    """
    Track all the crossings that happend previous to the current
    time.
    Swap the index i corresponding to the ith Molecular orbital
    with the corresponding swap at time t0.
    Repeat the same procedure with the index and swap at time t1.
    The swaps are compute with the linear sum assignment algorithm from Scipy.
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html

    """
    for i, mtx in enumerate(np.rollaxis(overlaps, 0)):
        overlaps[i] = mtx[:, swaps][swaps]

    return overlaps


def calculate_overlap(config: dict, mo_paths_hdf5: list) -> list:
    """
    Calculate the Overlap matrices before computing the non-adiabatic
    coupling using 3 consecutive set of MOs in a molecular dynamic.

    :returns: paths to the Overlap matrices inside the HDF5.
    """
    geometries = config.geometries
    nPoints = len(geometries) - 1

    def create_input_dict(i):
        return {
            'i': i,
            'molecules': select_molecules(config, geometries, i),
            'mo_paths': [mo_paths_hdf5[i + j][1] for j in range(2)]}

    comm = config.mpi_comm
    if comm is None or comm.Get_rank() == 0:
        paths = []
        for i in range(nPoints):
            overlaps_paths_hdf5 = create_overlap_path(config, i)
            overlaps, overlaps_paths_hdf5 = lazy_overlaps(
                config, overlaps_paths_hdf5, create_input_dict(i))
            store_arrays_in_hdf5(config.path_hdf5, overlaps_paths_hdf5, overlaps)
            paths.append(overlaps_paths_hdf5)

    # if comm is None or comm.Get_rank() == 0:
    #     paths = []
    #     rank = comm.Get_rank()
    #     size = comm.Get_size()
    #     for i in range(nPoints):
    #         # compute Overlaps
    #         overlaps_paths_hdf5 = create_overlap_path(config, i)
    #         residue = i % size
    #         if residue == rank:
    #             overlaps = lazy_overlaps(config, overlaps_paths_hdf5, create_input_dict(i))
    #             comm.Send((overlaps, overlaps_paths_hdf5), dest=0, tag=i)
    #         # Store array in HDF5
    #         if rank == 0:
    #             shape_array = 
    #             overlaps = np.empty(shape_array, dtype=np.float64)
    #             comm.Recv(overlaps, source=residue, tag=i)
    #             store_arrays_in_hdf5(config.path_hdf5, overlaps_paths_hdf5, overlaps)
    #         paths.append(overlaps_paths_hdf5)
        return paths
    else:
        return None


def select_molecules(config: dict, geometries: list, i: int):
    """
    Select the pairs of molecules to compute the couplings.
    """
    if config.overlaps_deph:
        return tuple(map(lambda idx: parse_string_xyz(geometries[idx]),
                         [0, i + 1]))
    else:
        return tuple(map(lambda idx: parse_string_xyz(geometries[idx]),
                         [i, i + 1]))


def lazy_overlaps(config: dict, overlaps_paths_hdf5: str,  dict_input: dict) -> str:
    """
    Calculate the 4 overlap matrix used to compute the subsequent couplings.
    The overlap matrices are computed using 3 consecutive set of MOs and
    3 consecutive geometries( in atomic units), from a molecular dynamics.

    :returns: path to the overlaps inside the HDF5
    """

    # If the Overlaps are not in the HDF5 file compute them
    if is_data_in_hdf5(config.path_hdf5, overlaps_paths_hdf5):
        logger.info("{} Overlaps are already in the HDF5".format(overlaps_paths_hdf5))
    else:
        # Read the Molecular orbitals from the HDF5
        logger.info("Computing: {}".format(overlaps_paths_hdf5))

        # Paths the MOs inside the HDF5
        overlaps = compute_overlaps_for_coupling(config, dict_input)

    return overlaps, overlaps_paths_hdf5


def create_overlap_path(config: dict, i: int) -> str:
    """
    Create the path inside the HDF5 where the overlap is going to be store.
    """
    root = join(config.project_name, 'overlaps_{}'.format(i + config.enumerate_from))
    return join(root, 'mtx_sji_t0')


def write_hamiltonians(config: dict, crossing_and_couplings: Tuple, mo_paths_hdf5: list) -> list:
    """
    Write the real and imaginary components of the hamiltonian using both
    the orbitals energies and the derivative coupling accoring to:
    http://pubs.acs.org/doi/abs/10.1021/ct400641n
    **Units are: Rydbergs**.
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
        energies_t1 = retrieve_hdf5_data(config.path_hdf5, mo_paths_hdf5[i + 1][0])

        # Return the average between time t and t + dt
        energies = np.average((energies_t0, energies_t1), axis=0)

        # Print Energies in the range given by the user
        if all(x is not None for x in [nHOMO, mo_index_range]):
            lowest = nHOMO - mo_index_range[0]
            highest = nHOMO + mo_index_range[1]
            energies = energies[lowest: highest]

        # Swap the energies of the states that are crossing
        energies = energies[swaps[i + 1]]

        # FileNames
        path_dir_results = config.path_hamiltonians
        file_ham_im = join(path_dir_results, 'Ham_{}_im'.format(j))
        file_ham_re = join(path_dir_results, 'Ham_{}_re'.format(j))

        # convert to Rydbergs
        ham_im = 2.0 * css
        ham_re = np.diag(2.0 * energies)

        write_pyxaid_format(ham_im, file_ham_im)
        write_pyxaid_format(ham_re, file_ham_re)

        return (file_ham_im, file_ham_re)

    # The couplings are compute at time t + dt therefore
    # we associate the energies at time t + dt with the corresponding coupling
    return [write_data(i) for i in range(config.nPoints)]


def swap_columns(arr: Matrix, swaps_t: Vector) -> Matrix:
    """
    Swap only columns at t+dt to reconstruct the original overlap matrix
    """
    return np.transpose(np.transpose(arr)[swaps_t])


def write_overlaps_in_ascii(overlaps: Tensor3D) -> None:
    """
    Write the corrected overlaps in text files.
    """
    if not os.path.isdir('overlaps'):
        os.mkdir('overlaps')

    # write overlaps
    nFrames = overlaps.shape[0]
    for k in range(nFrames):
        mtx_Sji = overlaps[k]
        path_Sji = 'overlaps/mtx_Sji_{}'.format(k)

        np.savetxt(path_Sji, mtx_Sji, fmt='%10.5e', delimiter='  ')
