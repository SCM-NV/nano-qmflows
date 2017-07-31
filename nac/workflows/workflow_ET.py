__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from .initialization import (
    create_map_index_pyxaid, read_time_dependent_coeffs)
from nac.schedule.components import calculate_mos
from nac.common import (
    Matrix, change_mol_units, femtosec2au,
    retrieve_hdf5_data, search_data_in_hdf5)
from nac.schedule.scheduleET import (
    compute_overlaps_ET, photo_excitation_rate)
from noodles import (gather, schedule)
from os.path import join
from qmworks import run
from qmworks.parsers import parse_string_xyz
from scipy import integrate

import logging
import numpy as np

from typing import (Dict, List, Tuple)

# Get logger
logger = logging.getLogger(__name__)

# ==============================> Main <==================================


def calculate_ETR(
        package_name: str, project_name: str, package_args: Dict,
        path_time_coeffs: str=None, geometries: List=None,
        initial_conditions: List=None, path_hdf5: str=None,
        basis_name: str=None,
        enumerate_from: int=0, package_config: Dict=None,
        calc_new_wf_guess_on_points: str=None, guess_args: Dict=None,
        work_dir: str=None, traj_folders: List=None,
        dictCGFs: Dict=None, orbitals_range: Tuple=None,
        pyxaid_HOMO: int=None, pyxaid_Nmin: int=None, pyxaid_Nmax: int=None,
        fragment_indices: None=List, dt: float=1, **kwargs):
    """
    Use a md trajectory to calculate the Electron transfer rate.

    :param package_name: Name of the package to run the QM simulations.
    :param project_name: Folder name where the computations
    are going to be stored.
    :param package_args: Specific settings for the package.
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param geometries: List of string cotaining the molecular geometries.
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :param calc_new_wf_guess_on_points: number of Computations that used a
                                        previous calculation as guess for the
                                        wave function.
    :param nHOMO: index of the HOMO orbital.
    :param couplings_range: Range of MO use to compute the nonadiabatic
    :param pyxaid_range: range of HOMOs and LUMOs used by pyxaid.
    :param enumerate_from: Number from where to start enumerating the folders
                           create for each point in the MD.
    :param traj_folders: List of paths to where the CP2K MOs are printed.
     :param package_config: Parameters required by the Package.
    :param fragment_indices: indices of atoms belonging to a fragment.
    :param dt: integration time used in the molecular dynamics.
    :returns: None
    """
    # prepare Cp2k Job point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(
        package_name, geometries, project_name, path_hdf5, traj_folders,
        package_args, guess_args, calc_new_wf_guess_on_points,
        enumerate_from, package_config=package_config)

    # geometries in atomic units
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for gs in geometries]

    # Time-dependent coefficients
    time_depend_coeffs = read_time_dependent_coeffs(path_time_coeffs)
    msg = "Reading time_dependent coefficients from: {}".format(path_time_coeffs)
    logger.info(msg)

    # compute_overlaps_ET
    scheduled_overlaps = schedule(compute_overlaps_ET)
    fragment_overlaps = scheduled_overlaps(
        project_name, molecules_au, basis_name, path_hdf5, dictCGFs,
        mo_paths_hdf5, fragment_indices, enumerate_from, package_name)

    # Delta time in a.u.
    dt_au = dt * femtosec2au

    # Indices relation between the PYXAID active space and the orbitals
    # stored in the HDF5
    map_index_pyxaid_hdf5 = create_map_index_pyxaid(
        orbitals_range, pyxaid_HOMO, pyxaid_Nmin, pyxaid_Nmax)

    # Number of ETR points calculated with the MD trajectory
    n_points = len(geometries) - 2

    # Read the swap between Molecular orbitals obtained from a previous
    # Coupling calculation
    swaps = read_swaps(path_hdf5, project_name)

    # Electron transfer rate for each frame of the Molecular dynamics
    scheduled_photoexcitation = schedule(compute_photoexcitation)
    etrs = scheduled_photoexcitation(
        path_hdf5, time_depend_coeffs, fragment_overlaps,
        map_index_pyxaid_hdf5, swaps, n_points, dt_au)

    # Execute the workflow
    electronTransferRates, path_overlaps = run(
        gather(etrs, fragment_overlaps), folder=work_dir)

    for i, mtx in enumerate(electronTransferRates):
        write_ETR(mtx, dt, i)

    write_overlap_densities(path_hdf5, path_overlaps, dt)


def compute_photoexcitation(
        path_hdf5: str, time_dependent_coeffs: Matrix,
        paths_fragment_overlaps: List, map_index_pyxaid_hdf5: Matrix,
        swaps: Matrix, n_points: int, dt_au: float) -> List:
    """
    :param i: Electron transfer rate at time i * dt
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param geometries: list of 3 contiguous molecular geometries
    :param time_depend_paths: mean value of the time dependent coefficients
    computed with PYXAID.
    param fragment_overlaps: Tensor containing 3 overlap matrices corresponding
    with the `geometries`.
    :param map_index_pyxaid_hdf5: map from PYXAID excitation to the indices i,j
    of the molecular orbitals stored in the HDF5.
    :param swaps: Matrix containing the crossing between the MOs during the
    molecular dynamics.
    :param n_points: Number of frames to compute the ETR.
    :param dt_au: Delta time in atomic units
    :returns: promise to path to the Coupling inside the HDF5.
    """
    msg = "Computing the photo-excitation rate for the molecular fragments"
    logger.info(msg)

    results = []
    for paths_overlaps in paths_fragment_overlaps:
        overlaps = np.stack(retrieve_hdf5_data(path_hdf5, paths_overlaps))
        # Track the crossing between MOs
        for k, mtx in enumerate(np.rollaxis(overlaps, 0)):
            overlaps[k] = mtx[:, swaps[k]][swaps[k]]

        etr = np.array([
            photo_excitation_rate(
                overlaps[i: i + 3], time_dependent_coeffs[i: i + 3],
                map_index_pyxaid_hdf5, dt_au)
            for i in range(n_points)])
        results.append(etr)

    return np.stack(results)


def read_swaps(path_hdf5: str, project_name: str) -> Matrix:
    """
    Read the crossing tracking for the Molecular orbital
    """
    path_swaps = join(project_name, 'swaps')
    if search_data_in_hdf5(path_hdf5, path_swaps):
        return retrieve_hdf5_data(path_hdf5, path_swaps)
    else:
        msg = """There is not a tracking file called: {}
        This file is automatically created when running the worflow_coupling
        simulations""".format(path_swaps)
        raise RuntimeError(msg)


def write_ETR(mtx, dt, i):
    """
    Save the ETR in human readable format
    """
    file_name = "electronTranferRates_fragment_{}.txt".format(i)
    header = 'electron_Density electron_ETR(Nonadiabatic Adibatic) hole_density hole_ETR(Nonadiabatic Adibatic)'
    # Density of Electron/hole
    density_electron = mtx[1:, 0]
    density_hole = mtx[1:, 3]
    # Integrate the nonadiabatic/adiabatic components of the electron/hole ETR
    int_elec_nonadia, int_elec_adia, int_hole_nonadia, int_hole_adia = [
        integrate.cumtrapz(mtx[:, k], dx=dt) for k in [1, 2, 4, 5]]

    # Join the data
    data = np.stack(
        (density_electron, int_elec_nonadia, int_elec_adia, density_hole,
         int_hole_nonadia, int_hole_adia), axis=1)

    # save the data in human readable format
    np.savetxt(file_name, data, fmt='{:^3}'.format('%e'), header=header)


def write_overlap_densities(
        path_hdf5: str, paths_fragment_overlaps: List, dt: int=1):
    """
    Write the diagonal of the overlap matrices
    """
    logger.info("writing densities in human readable format")
    for k, paths_overlaps in enumerate(paths_fragment_overlaps):
        overlaps = np.stack(retrieve_hdf5_data(path_hdf5, paths_overlaps))
        # time frame
        frames = overlaps.shape[0]
        ts = np.arange(1, frames + 1).reshape(frames, 1) * dt
        # Diagonal of the 3D-tensor
        densities = np.diagonal(overlaps, axis1=1, axis2=2)
        data = np.hstack((ts, densities))

        # Save data in human readable format
        file_name = 'densities_fragment_{}.txt'.format(k)
        np.savetxt(file_name, data, fmt='{:^3}'.format('%e'))
