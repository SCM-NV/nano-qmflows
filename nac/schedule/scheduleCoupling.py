__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from os.path import join
from scipy.optimize import linear_sum_assignment

import logging
import numpy as np
import os
# ==================> Internal modules <==========
from nac.integrals import (
    calculate_couplings_levine, calculate_couplings_3points,
    compute_overlaps_for_coupling, correct_phases)
from nac.common import (
    Matrix, Vector, Tensor3D,
    change_mol_units, femtosec2au, retrieve_hdf5_data,
    search_data_in_hdf5, store_arrays_in_hdf5)

from noodles import (gather, schedule)
from qmworks.parsers import parse_string_xyz

# Types hint
from typing import (Callable, Dict, List, Tuple)

# Starting logger
logger = logging.getLogger(__name__)

# ==============================> Schedule Tasks <=============================


def lazy_couplings(paths_overlaps: List, path_hdf5: str, project_name: str,
                   enumerate_from: int, nHOMO: int, dt: float, tracking: bool,
                   algorithm='levine') -> List:
    """
    Compute the Nonadibatic coupling using a 3 point approximation. See:
    The Journal of Chemical Physics 137, 22A514 (2012); doi: 10.1063/1.4738960

    or a 2Point approximation using an smoothing function:
    J. Phys. Chem. Lett. 2014, 5, 2351−2356; doi: 10.1021/jz5009449

    Notice that the states can cross frequently due to unavoided crossing and
    such crossing must be track. see:
    J. Chem. Phys. 137, 014512 (2012); doi: 10.1063/1.4732536
    """
    if tracking:
        fixed_phase_overlaps, swaps = compute_the_fixed_phase_overlaps(
            paths_overlaps, path_hdf5, project_name, enumerate_from, nHOMO)
    else:
        # Do not track the crossings
        fixed_phase_overlaps = np.stack(
            retrieve_hdf5_data(path_hdf5, paths_overlaps))
        nOverlaps, nOrbitals, _ = fixed_phase_overlaps.shape
        swaps = np.tile(np.arange(nOrbitals), (nOverlaps + 1, 1))

    # Write the overlaps in text format
    logger.debug("Writing down the overlaps in ascii format")
    write_overlaps_in_ascii(fixed_phase_overlaps)

    # Compute the couplings using either the levine method
    # or the 3Points approximation
    coupling_algorithms = {'levine': (calculate_couplings_levine, 1),
                           '3points': (calculate_couplings_3points, 2)}
    # Choose an algorithm to compute the couplings
    fun_coupling, step = coupling_algorithms[algorithm]

    # Number of couplings to compute
    nCouplings = fixed_phase_overlaps.shape[0] - step + 1

    # time in atomic units
    dt_au = dt * femtosec2au

    couplings = [calculate_couplings(
        fun_coupling, algorithm, i, project_name, fixed_phase_overlaps,
        path_hdf5, enumerate_from, dt_au) for i in range(nCouplings)]

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
    if not search_data_in_hdf5(path_hdf5, paths_corrected_overlaps[0]):

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


def calculate_couplings(
        fun_coupling: Callable, algorithm: str, i: int,
        project_name: str, fixed_phase_overlaps: Tensor3D,
        path_hdf5: str, enumerate_from: int, dt_au: float) -> str:
    """
    Search for the ith Coupling in the HDF5, if it is not available compute it
    using the 3 points approximation.
    """
    # Path were the couplinp is store
    k = i + enumerate_from
    path = join(project_name, 'coupling_{}'.format(k))

    # Skip the computation if the coupling is already done
    if search_data_in_hdf5(path_hdf5, path):
        logger.info("Coupling: {} has already been calculated".format(path))
        return path
    else:
        logger.info("Computing coupling: {}".format(path))
        if algorithm == 'levine':
            # Extract the overlap matrices involved in the coupling computation
            sji_t0 = fixed_phase_overlaps[i]
            # Compute the couplings with the phase corrected overlaps
            couplings = fun_coupling(dt_au, sji_t0, sji_t0.transpose())
        elif algorithm == '3points':
            sji_t0, sji_t1 = fixed_phase_overlaps[i: i + 2]
            couplings = fun_coupling(dt_au, sji_t0, sji_t0.transpose(), sji_t1,
                                     sji_t1.transpose())

            # Store the Coupling in the HDF5
        store_arrays_in_hdf5(path_hdf5, path, couplings)

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


def calculate_overlap(project_name: str, path_hdf5: str, dictCGFs: Dict,
                      geometries: List, mo_paths_hdf5: List,
                      hdf5_trans_mtx: str, enumerate_from: int,
                      nHOMO: int=None, couplings_range: Tuple=None,
                      units: str='angstrom') -> List:
    """
    Calculate the Overlap matrices before computing the non-adiabatic
    coupling using 3 consecutive set of MOs in a molecular dynamic.

    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :type path_hdf5: String
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :param geometries: list of molecular geometries
    :param mo_paths: Path to the MO coefficients and energies in the
    HDF5 file.
    :param hdf5_trans_mtx: path to the transformation matrix in the HDF5 file.
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :type enumerate_from: Int
    :param nHOMO: index of the HOMO orbital in the HDF5
    :param couplings_range: range of Molecular orbitals used to compute the
    coupling.
    :returns: paths to the Overlap matrices inside the HDF5.
    """
    nPoints = len(geometries) - 1

    # Inplace scheduling of calculate_overlap function
    # Equivalent to add @schedule on top of the function
    schedule_overlaps = schedule(lazy_overlaps)

    # Compute the Overlaps
    paths_overlaps = []
    for i in range(nPoints):

        # extract 3 molecular geometries to compute the overlaps
        molecules = tuple(map(lambda idx: parse_string_xyz(geometries[idx]),
                              [i, i + 1]))

        # If units are Angtrom convert then to a.u.
        if 'angstrom' in units.lower():
            molecules = tuple(map(change_mol_units, molecules))

        # Compute the coupling
        overlaps = schedule_overlaps(
            i, project_name, path_hdf5, dictCGFs, molecules, mo_paths_hdf5,
            hdf5_trans_mtx=hdf5_trans_mtx, enumerate_from=enumerate_from,
            nHOMO=nHOMO, couplings_range=couplings_range)

        paths_overlaps.append(overlaps)

    # Gather all the promised paths
    return gather(*paths_overlaps)

def lazy_overlaps(i: int, project_name: str, path_hdf5: str, dictCGFs: Dict,
                  geometries: Tuple, mo_paths: List, hdf5_trans_mtx: str=None,
                  enumerate_from: int=0, nHOMO: int=None,
                  couplings_range: Tuple=None) -> str:
    """
    Calculate the 4 overlap matrix used to compute the subsequent couplings.
    The overlap matrices are computed using 3 consecutive set of MOs and
    3 consecutive geometries( in atomic units), from a molecular dynamics.

    :param i: nth coupling calculation
    :param project_name: Name of the project to be executed.
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :parameter geometries: molecular geometries stored as list of
                           namedtuples.
    :type      geometries: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :parameter mo_paths: List of paths to the MO in the HDF5
    :param hdf5_trans_mtx: Path to the transformation matrix in the HDF5
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :param nHOMO: index of the HOMO orbital in the HDF5
    :param couplings_range: range of Molecular orbitals used to compute the
    coupling.

    :returns: path to the Coupling inside the HDF5
    """
    # Path inside the HDF5 where the overlaps are stored
    root = join(project_name, 'overlaps_{}'.format(i + enumerate_from))
    overlaps_paths_hdf5 = join(root, 'mtx_sji_t0')

    # If the Overlaps are not in the HDF5 file compute them
    if search_data_in_hdf5(path_hdf5, overlaps_paths_hdf5):
        logger.info("{} Overlaps are already in the HDF5".format(root))
    else:
        # Read the Molecular orbitals from the HDF5
        logger.info("Computing: {}".format(root))

        # Paths to the MOs inside the HDF5
        hdf5_mos_path = [mo_paths[i + j][1] for j in range(2)]

        # Partial application of the function computing the overlap
        overlaps = compute_overlaps_for_coupling(
            geometries, path_hdf5, hdf5_mos_path, dictCGFs, nHOMO,
            couplings_range, hdf5_trans_mtx)

        # Store the matrices in the HDF5 file
        store_arrays_in_hdf5(path_hdf5, overlaps_paths_hdf5, overlaps)

    return overlaps_paths_hdf5


def write_hamiltonians(path_hdf5: str, mo_paths: List,
                       crossing_and_couplings: Tuple,
                       nPoints: int, path_dir_results: str=None,
                       enumerate_from: int=0, nHOMO: int=None,
                       couplings_range: Tuple=None) -> None:
    """
    Write the real and imaginary components of the hamiltonian using both
    the orbitals energies and the derivative coupling accoring to:
    http://pubs.acs.org/doi/abs/10.1021/ct400641n
    **Units are: Rydbergs**.
    """
    swaps, path_couplings = crossing_and_couplings

    def write_pyxaid_format(arr, fileName):
        np.savetxt(fileName, arr, fmt='%10.5e', delimiter='  ')

    def write_data(i):
        j = i + enumerate_from
        path_coupling = path_couplings[i]
        css = retrieve_hdf5_data(path_hdf5, path_coupling)

        # Extract the energy values at time t
        # The first coupling is compute at time t + dt
        # Then I'm shifting the energies dt to get the correct value
        energies_t0 = retrieve_hdf5_data(path_hdf5, mo_paths[i][0])
        energies_t1 = retrieve_hdf5_data(path_hdf5, mo_paths[i + 1][0])

        # Return the average between time t and t + dt
        energies = np.average((energies_t0, energies_t1), axis=0)

        # Print Energies in the range given by the user
        if all(x is not None for x in [nHOMO, couplings_range]):
            lowest = nHOMO - couplings_range[0]
            highest = nHOMO + couplings_range[1]
            energies = energies[lowest: highest]

        # Swap the energies of the states that are crossing
        energies = energies[swaps[i + 1]]

        # FileNames
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
    return [write_data(i) for i in range(nPoints)]


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
