"""Module to run restricted and unrestricted calculations."""

import logging
from typing import Any, Callable

from noodles import gather
from qmflows import run

from ..common import DictConfig
from .initialization import initialize

# Starting logger
logger = logging.getLogger(__name__)


def select_orbitals_type(
        config: DictConfig, workflow: Callable[[DictConfig], Any]) -> Any:
    """Call a workflow using restriced or unrestricted orbitals."""
    # Dictionary containing the general configuration
    config.update(initialize(config))

    if config.orbitals_type != "both":
        logger.info("starting workflow calculation!")
        promises = workflow(config)
        return run(promises, folder=config.workdir, always_cache=False)
    else:
        config_alphas = DictConfig(config.copy())
        config_betas = DictConfig(config.copy())
        config_alphas.orbitals_type = "alphas"
        promises_alphas = workflow(config_alphas)
        config_betas.orbitals_type = "betas"
        promises_betas = workflow(config_betas)
        all_promises = gather(promises_alphas, promises_betas)
        alphas, betas = run(all_promises, folder=config.workdir, always_cache=False)
        return alphas, betas
