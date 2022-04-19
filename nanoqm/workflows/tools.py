"""Common utilities use by the workflows."""

from .. import logger
from ..common import DictConfig, is_data_in_hdf5
from .workflow_single_points import workflow_single_points


def compute_single_point_eigenvalues_coefficients(config: DictConfig):
    """Check if hdf5 contains the required eigenvalues and coefficients.

    If not, it runs the single point calculation.
    """
    node_path_coefficients = f'{config.project_name}/point_0/cp2k/mo/coefficients'
    node_path_eigenvalues = f'{config.project_name}/point_0/cp2k/mo/eigenvalues'

    node_paths = (node_path_coefficients, node_path_eigenvalues)
    if all(is_data_in_hdf5(config.path_hdf5, x) for x in node_paths):
        logger.info("Coefficients and eigenvalues already in hdf5.")
    else:
        # Call the single point workflow to calculate the eigenvalues and
        # coefficients
        logger.info("Starting single point calculation.")
        workflow_single_points(config)
