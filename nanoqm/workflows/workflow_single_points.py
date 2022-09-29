"""Workflow to perform single point calculation in a trajectory.

Index
-----
.. currentmodule:: nanoqm.workflows.workflow_single_points
.. autosummary::
    workflow_single_points

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qmflows import run

from .. import logger
from ..schedule.components import calculate_mos
from .initialization import initialize

if TYPE_CHECKING:
    from .. import _data

__all__ = ['workflow_single_points']


def workflow_single_points(
    config: _data.SinglePoints,
) -> tuple[list[tuple[str, str, str]], list[str]]:
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
    initialize(config)

    logger.info("starting!")

    # compute the molecular orbitals
    # Unpack
    mo_paths_hdf5 = calculate_mos(config)

    # Pack
    return tuple(run(mo_paths_hdf5, folder=config.workdir))
