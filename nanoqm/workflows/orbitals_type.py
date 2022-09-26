"""Module to run restricted and unrestricted calculations."""

from __future__ import annotations

import copy
from typing import Any, Callable, TypeVar, TYPE_CHECKING

from noodles import gather
from qmflows import run

from .. import logger
from .initialization import initialize

if TYPE_CHECKING:
    from qmflows.type_hints import PromisedObject
    from .. import _data

    _T = TypeVar("_T", bound=_data.GeneralOptions)

__all__ = ["select_orbitals_type"]


def select_orbitals_type(config: _T, workflow: Callable[[_T], PromisedObject]) -> Any:
    """Call a workflow using restriced or unrestricted orbitals."""
    # Dictionary containing the general configuration
    initialize(config)

    if config.orbitals_type != "both":
        logger.info("starting workflow calculation!")
        promises = workflow(config)
        return run(promises, folder=config.workdir, always_cache=False)
    else:
        config_alphas = copy.copy(config)
        config_betas = copy.copy(config)
        config_alphas.orbitals_type = "alphas"
        promises_alphas = workflow(config_alphas)
        config_betas.orbitals_type = "betas"
        promises_betas = workflow(config_betas)
        all_promises = gather(promises_alphas, promises_betas)
        alphas, betas = run(all_promises, folder=config.workdir, always_cache=False)
        return alphas, betas
