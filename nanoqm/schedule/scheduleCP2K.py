"""Module to configure and run CP2K jobs.

Index
-----
.. currentmodule:: nanoqm.schedule.scheduleCP2K
.. autosummary::
    prepare_job_cp2k

API
---
.. autofunction:: prepare_job_cp2k

"""

import fnmatch
import logging
import os
from os.path import join
from pathlib import Path
from typing import Any, Dict

from noodles import schedule  # Workflow Engine
from qmflows import Settings, cp2k, templates
from qmflows.packages.cp2k_package import CP2K, CP2K_Result
from qmflows.parsers.xyzParser import string_to_plams_Molecule
from qmflows.type_hints import PathLike, PromisedObject

# Starting logger
logger = logging.getLogger(__name__)


def try_to_read_wf(path_dir: PathLike) -> PathLike:
    """Try to get a wave function file from ``path_dir``.

    Returns
    -------
    str
        Path to the wave function file.

    Raises
    ------
    RuntimeError
        If there is not a wave function file.

    """
    xs = os.listdir(path_dir)
    files = list(filter(lambda x: fnmatch.fnmatch(x, '*wfn'), xs))
    if files:
        return join(path_dir, files[0])
    else:
        msg = f"There are no wave function file in path:{path_dir}\n"
        msg += print_cp2k_error(path_dir, "err")
        msg += print_cp2k_error(path_dir, "out")
        raise RuntimeError(msg)


def prepare_cp2k_settings(
        settings: Settings, dict_input: Dict[str, Any], guess_job: CP2K_Result) -> Settings:
    """Fill in the parameters for running a single job in CP2K.

    Parameters
    ----------
    settings
        Input for CP2K
    dict_input
        Input for the current molecular geometry
    guess_job
        Previous job to read the guess wave function

    Returns
    .......
    CP2K
        job to run

    """
    dft = settings.specific.cp2k.force_eval.dft
    dft['print']['mo']['filename'] = dict_input["job_files"].get_MO

    # Global parameters for CP2K
    settings.specific.cp2k['global']['project'] = f'point_{dict_input["k"]}'

    if guess_job is not None:
        dft.wfn_restart_file_name = try_to_read_wf(
            guess_job.archive['plams_dir'])

    input_args = templates.singlepoint.overlay(settings)

    return input_args


@schedule
def prepare_job_cp2k(
        settings: Settings, dict_input: Dict[str, Any], guess_job: PromisedObject) -> CP2K:
    """Generate a :class:`qmflows.packages.cp2k_packages.CP2K` job.

    Parameters
    ----------
    settings
        Input for CP2K
    dict_input
        Input for the current molecular geometry
    guess_job
        Previous job to read the guess wave function

    Returns
    -------
    :class:`qmflows.packages.cp2k_package.CP2K`
        job to run

    """
    job_settings = prepare_cp2k_settings(settings, dict_input, guess_job)

    # remove keywords not use on the next translation phase
    for x in ('basis', 'potential'):
        if x in job_settings:
            del job_settings[x]

    return cp2k(
        job_settings, string_to_plams_Molecule(dict_input["geometry"]),
        work_dir=dict_input['point_dir'])


def print_cp2k_error(path_dir: PathLike, prefix: str) -> str:
    """Search for error in the CP2K output files."""
    err_file = next(Path(path_dir).glob(f"*{prefix}"), None)
    msg = ""
    if err_file is not None:
        with open(err_file, 'r') as handler:
            err = handler.read()
        msg = f"CP2K {prefix} file:\n{err}\n"
        logger.error(msg)

    return msg
