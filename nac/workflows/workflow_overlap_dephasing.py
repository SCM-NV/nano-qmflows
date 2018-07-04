__author__ = "Ivan Infante"

__all__ = ['generate_overlap_dephasing']

# ================> Python Standard  and third-party <==========

from .initialization import log_config
from nac.schedule.components import calculate_mos
from nac.schedule.scheduleCoupling import (
    calculate_overlap_dephasing, write_overlaps_in_ascii)
from noodles import schedule
from os.path import join
from qmworks import run

import logging
import os
import shutil

# Type Hints
from typing import (Dict, List, Tuple)

# ==============================> Main <==================================


def generate_overlap_dephasing(
        package_name: str, project_name: str,
        package_args: Dict, guess_args: Dict=None,
        geometries: List=None, dictCGFs: Dict=None,
        calc_new_wf_guess_on_points: str=None,
        path_hdf5: str=None, enumerate_from: int=0,
        package_config: Dict=None, dt: float=1,
        traj_folders: List=None, work_dir: str=None,
        basisname: str=None, hdf5_trans_mtx: str=None,
        nHOMO: int=None, couplings_range: Tuple=None,
        algorithm='levine', ignore_warnings=False, tracking=True) -> None:
    """
    Use a MD trajectory to compute the overlap dephasing 

    :param package_name: Name of the package to run the QM simulations.
    :param project_name: Folder name where the computations
    are going to be stored.
    :param package_args: Specific Settings for the package that will compute
    the MOs.
    :param geometries: List of string cotaining the molecular geometries
    numerical results.
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :param calc_new_wf_guess_on_points: Calculate a guess wave function either
    in the first point or on each point of the trajectory.
    :param path_hdf5: path to the HDF5 file were the data is going to be stored.
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD.
    :param hdf5_trans_mtx: Path into the HDF5 file where the transformation
    matrix (from Cartesian to sphericals) is stored.
    :param dt: Time used in the dynamics (femtoseconds)
    :param package_config: Parameters required by the Package.
    :type package_config: Dict
    :param nHOMO: index of the HOMO orbital.
    :param couplings_range: Range of MO use to compute the nonadiabatic
    coupling matrix.

    :returns: None
    """
    # Log initial config information
    log_config(work_dir, path_hdf5, algorithm)

    # prepare Cp2k Jobs
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(
        package_name, geometries, project_name, path_hdf5, traj_folders,
        package_args, guess_args, calc_new_wf_guess_on_points,
        enumerate_from, package_config=package_config,
        ignore_warnings=ignore_warnings)

    # Overlap matrix at two different times
    promised_overlaps_dephasing = calculate_overlap_dephasing(
        project_name, path_hdf5, dictCGFs, geometries, mo_paths_hdf5,
        hdf5_trans_mtx, enumerate_from, nHOMO=nHOMO,
        couplings_range=couplings_range)

    # Write the overlaps in text format
    logger.debug("Writing down the overlaps in ascii format")
    write_overlaps_in_ascii(promised_overlaps_dephasing)
   
    # Remove folders
    remove_folders(traj_folders)

def remove_folders(folders):
    """
    Remove unused folders
    """
    for f in folders:
        if os.path.exists(f):
            shutil.rmtree(f)
