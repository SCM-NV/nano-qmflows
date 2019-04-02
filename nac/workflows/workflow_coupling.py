
__all__ = ['workflow_derivative_couplings']


from nac.schedule.components import calculate_mos
from nac.schedule.scheduleCoupling import (
    calculate_overlap, lazy_couplings, write_hamiltonians)
from nac.workflows.initialization import initialize
from noodles import schedule
from os.path import join
from qmflows import run

import logging
import os
import shutil

# Starting logger
logger = logging.getLogger(__name__)


def workflow_derivative_couplings(config: dict) -> list:
    """
    Compute the derivative couplings from an MD trajectory.

    :param workflow_settings: Arguments to compute the oscillators see:
    `nac/workflows/schemas.py
    :returns: None
    """
    # Dictionary containing the general configuration
    config.update(initialize(config))

    logger.info("starting!")

    # compute the molecular orbitals
    # mo_paths_hdf5 = calculate_mos(config)

    mo_paths_hdf5 = run(calculate_mos(config), folder=config.workdir)

    # Overlap matrix at two different times
    promised_overlaps = calculate_overlap(config, mo_paths_hdf5)

    # Calculate Non-Adiabatic Coupling
    promised_crossing_and_couplings = lazy_couplings(config, promised_overlaps)

    # Write the results in PYXAID format
    config.path_hamiltonians = create_path_hamiltonians(config.workdir)

    # Inplace scheduling of write_hamiltonians function.
    # Equivalent to add @schedule on top of the function
    schedule_write_ham = schedule(write_hamiltonians)

    # Number of matrix computed
    config["nPoints"] = len(config.geometries) - 2

    # Write Hamilotians in PYXAID format
    promise_files = schedule_write_ham(
        config, promised_crossing_and_couplings, mo_paths_hdf5)

    results = run(promise_files, folder=config.workdir)

    remove_folders(config.folders)

    return results


def create_path_hamiltonians(workdir: str) -> str:
    """ Path to store the resulting hamiltonians """
    path_hamiltonians = join(workdir, 'hamiltonians')
    if not os.path.exists(path_hamiltonians):
        os.makedirs(path_hamiltonians)

    return path_hamiltonians


def remove_folders(folders):
    """
    Remove unused folders
    """
    for f in folders:
        if os.path.exists(f):
            shutil.rmtree(f)
