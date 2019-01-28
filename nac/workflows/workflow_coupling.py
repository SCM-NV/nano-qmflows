
__all__ = ['workflow_derivative_couplings']


from nac.schedule.components import calculate_mos
from nac.schedule.scheduleCoupling import (
    calculate_overlap, lazy_couplings, write_hamiltonians)
from nac.workflows.initialization import initialize
from noodles import schedule
from os.path import join
from qmflows import run

import os
import shutil

# Type Hints
from typing import Dict


def workflow_derivative_couplings(workflow_settings: Dict):
    """
    Compute the derivative couplings from an MD trajectory.

    :param workflow_settings: Arguments to compute the oscillators see:
    `data/schemas/derivative_couplings.json
    :returns: None
    """
    # Arguments to compute the orbitals and configure the workflow. see:
    # `data/schemas/general_settings.json
    config = workflow_settings['general_settings']
    print("config: ", config)

    # Dictionary containing the general configuration
    config.update(initialize(**config))

    # compute the molecular orbitals
    mo_paths_hdf5 = calculate_mos(**config)

    # Overlap matrix at two different times
    promised_overlaps = calculate_overlap(
        config['project_name'], config['path_hdf5'], config['dictCGFs'],
        config['geometries'], mo_paths_hdf5,
        config['hdf5_trans_mtx'], config['enumerate_from'],
        workflow_settings['overlaps_deph'], nHOMO=config['nHOMO'],
        mo_index_range=config['mo_index_range'])

    # Create a function that returns a proxime array of couplings
    schedule_couplings = schedule(lazy_couplings)

    # Calculate Non-Adiabatic Coupling
    promised_crossing_and_couplings = schedule_couplings(
        promised_overlaps, config['path_hdf5'], config['project_name'],
        config['enumerate_from'], config['nHOMO'],
        workflow_settings['dt'], workflow_settings['tracking'],
        workflow_settings['write_overlaps'],
        algorithm=workflow_settings['algorithm'])

    # Write the results in PYXAID format
    work_dir = config['work_dir']
    path_hamiltonians = join(work_dir, 'hamiltonians')
    if not os.path.exists(path_hamiltonians):
        os.makedirs(path_hamiltonians)

    # Inplace scheduling of write_hamiltonians function.
    # Equivalent to add @schedule on top of the function
    schedule_write_ham = schedule(write_hamiltonians)

    # Number of matrix computed
    nPoints = len(config['geometries']) - 2

    # Write Hamilotians in PYXAID format
    promise_files = schedule_write_ham(
        config['path_hdf5'], mo_paths_hdf5, promised_crossing_and_couplings,
        nPoints, path_dir_results=path_hamiltonians,
        enumerate_from=config['enumerate_from'], nHOMO=config['nHOMO'],
        mo_index_range=config['mo_index_range'])

    run(promise_files, folder=work_dir)

    remove_folders(config['folders'])


def remove_folders(folders):
    """
    Remove unused folders
    """
    for f in folders:
        if os.path.exists(f):
            shutil.rmtree(f)
