"""Workflow to perform single point calculation in a trajectory.

Index
-----
.. currentmodule:: nanoqm.workflows.workflow_single_points
.. autosummary::
    workflow_single_points

"""

__all__ = ['workflow_single_points']


import logging
from typing import List

from qmflows import run
from qmflows.packages import Result

from ..common import DictConfig
from ..schedule.components import calculate_mos
from .initialization import initialize

# Starting logger
logger = logging.getLogger(__name__)


def workflow_single_points(config: DictConfig) -> Result:
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
