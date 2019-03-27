
__all__ = ['workflow_single_points']


from nac.schedule.components import calculate_mos
from nac.workflows.initialization import initialize
from qmflows import run

import logging


# Starting logger
logger = logging.getLogger(__name__)


def workflow_single_points(config: dict) -> list:
    """
    Single point calculations for a given trajectory

    :param workflow_settings: Arguments to run the single points calculations see:
    `nac/workflows/schemas.py
    """
    # Dictionary containing the general configuration
    config.update(initialize(config))

    logger.info("starting!")

    # compute the molecular orbitals
    mo_paths_hdf5 = calculate_mos(config)

    results = run(mo_paths_hdf5, folder=config.workdir)

    return results
