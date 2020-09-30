"""Workflow to compute the derivate coupling between states.

The ``workflow_derivative_couplings`` expected a file with a trajectory-like
file with the molecular geometries to compute the couplings.

Index
-----
.. currentmodule:: nanoqm.workflows.workflow_coupling
.. autosummary::

"""
__all__ = ['workflow_derivative_couplings']


import logging
import os
from os.path import join
from typing import List, Tuple, Union

from noodles import gather, schedule, unpack
from noodles.interface import PromisedObject
from qmflows import run
from qmflows.type_hints import PathLike

from ..common import DictConfig
from ..schedule.components import calculate_mos
from ..schedule.scheduleCoupling import (calculate_overlap, lazy_couplings,
                                         write_hamiltonians)
from .initialization import initialize

# Starting logger
logger = logging.getLogger(__name__)

#: Type defining the derivative couplings calculation
ResultPaths = Tuple[List[str], List[str]]


def workflow_derivative_couplings(config: DictConfig) -> Union[ResultPaths, Tuple[ResultPaths, ResultPaths]]:
    """Compute the derivative couplings for a molecular dynamic trajectory."""
    # Dictionary containing the general configuration
    config.update(initialize(config))

    if config.orbitals_type != "both":
        logger.info("starting couplings calculation!")
        promises = run_workflow_couplings(config)
        return run(promises, folder=config.workdir, always_cache=False)
    else:
        config_alphas = DictConfig(config.copy())
        config_betas = DictConfig(config.copy())
        config_alphas.orbitals_type = "alphas"
        promises_alphas = run_workflow_couplings(config_alphas)
        config_betas.orbitals_type = "betas"
        promises_betas = run_workflow_couplings(config_betas)
        all_promises = gather(promises_alphas, promises_betas)
        alphas, betas = run(all_promises, folder=config.workdir, always_cache=False)
        return alphas, betas


def run_workflow_couplings(config: DictConfig) -> PromisedObject:
    """Run the derivative coupling workflow using `config`."""
    # compute the molecular orbitals
    logger.info("starting couplings calculation!")
    mo_paths_hdf5, energy_paths_hdf5 = unpack(calculate_mos(config), 2)

    # Overlap matrix at two different times
    promised_overlaps = calculate_overlap(config, mo_paths_hdf5)

    # Calculate Non-Adiabatic Coupling
    promised_crossing_and_couplings = lazy_couplings(config, promised_overlaps)

    # Write the results in PYXAID format
    config.path_hamiltonians = create_path_hamiltonians(config.workdir, config.orbitals_type)

    # Inplace scheduling of write_hamiltonians function.
    # Equivalent to add @schedule on top of the function
    schedule_write_ham = schedule(write_hamiltonians)

    # Number of matrix computed
    config["npoints"] = len(config.geometries) - 2

    # Write Hamilotians in PYXAID format
    promise_files = schedule_write_ham(
        config, promised_crossing_and_couplings, mo_paths_hdf5)

    return gather(promise_files, energy_paths_hdf5)


def create_path_hamiltonians(workdir: PathLike, orbitals_type: str) -> PathLike:
    """Create the Paths to store the resulting hamiltonians."""
    prefix = "hamiltonians"
    name = prefix if not orbitals_type else f"{orbitals_type}_{prefix}"
    path_hamiltonians = join(workdir, name)
    if not os.path.exists(path_hamiltonians):
        os.makedirs(path_hamiltonians)

    return path_hamiltonians
