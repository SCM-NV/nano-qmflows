"""Workflow to perform single point calculation in a trajectory.

Index
-----
.. currentmodule:: nac.workflows.workflow_single_points
.. autosummary::
    workflow_single_points

"""

__all__ = ['workflow_single_points']


import logging

from qmflows import run
from qmflows.type_hints import PathLike

from ..common import DictConfig
from ..schedule.components import calculate_mos
from .initialization import initialize
from typing import List

# Starting logger
logger = logging.getLogger(__name__)


def workflow_single_points(config: DictConfig) -> List[PathLike]:
    """Perform single point calculations for a given trajectory.

    Parameters
    ----------
    config
        Input to run the workflow.

    Returns
    -------
    List with the node path to the molecular orbitals in the HDF5.

    """
    # Dictionary containing the general configuration
    config.update(initialize(config))

    logger.info("starting!")

    # compute the molecular orbitals
    # Unpack
    mo_paths_hdf5 = calculate_mos(config)
    # Pack
    results = run(mo_paths_hdf5, folder=config.workdir)

    return results
