"""Molecular orbitals calculation using CP2K and `QMFlows <https://github.com/SCM-NV/qmflows>`.

API
---
.. autofunction:: calculate_mos

"""

from __future__ import annotations

__all__ = ["calculate_mos", "create_point_folder",
           "split_file_geometries"]

import fnmatch
import os
from collections import defaultdict
from os.path import join
from typing import (Any, DefaultDict, Dict, List, NamedTuple, Sequence, Tuple,
                    Union)

import numpy as np
from noodles import gather, schedule
from qmflows.common import CP2KInfoMO
from qmflows.type_hints import PathLike, PromisedObject
from qmflows.warnings_qmflows import SCF_Convergence_Warning

from .. import logger
from ..common import (DictConfig, Matrix, is_data_in_hdf5,
                      read_cell_parameters_as_array, store_arrays_in_hdf5)
from .scheduleCP2K import prepare_job_cp2k

#: Molecular orbitals from both restricted and unrestricted calculations
OrbitalType = Union[CP2KInfoMO, Tuple[CP2KInfoMO, CP2KInfoMO]]


class JobFiles(NamedTuple):
    """Contains the data to compute the molecular orbitals for a given geometry."""

    get_xyz: PathLike
    get_inp: PathLike
    get_out: PathLike
    get_MO: PathLike


def calculate_mos(config: DictConfig) -> List[str]:
    """Look for the MO in the HDF5 file and compute them if they are not present.

    The orbitals are computed  by splitting the jobs in batches given by
    the ``restart_chunk`` variables. Only the first job is calculated from scratch
    while the rest of the batch uses as guess the wave function of the first calculation
    inthe batch.

    The config dict contains:
        * geometries: list of molecular geometries
        * project_name: Name of the project used as root path for storing data in HDF5.
        * path_hdf5: Path to the HDF5 file that contains the numerical results.
        * folders: path to the directories containing the MO outputs
        * settings_main: Settings for the job to run.
        * calc_new_wf_guess_on_points: Calculate a new Wave function guess in each of the geometries indicated. By Default only an initial guess is computed.
        * enumerate_from: Number from where to start enumerating the folders create for each point in the MD

    Returns
    -------
        paths to the datasets in the HDF5 file containging both the MO energies and MO coefficients

    """
    # Read Cell parameters file
    general = config['cp2k_general_settings']
    file_cell_parameters = general["file_cell_parameters"]
    if file_cell_parameters is not None:
        array_cell_parameters = read_cell_parameters_as_array(file_cell_parameters)[
            1]

    # First calculation has no initial guess
    # calculate the rest of the jobs using the previous point as initial guess
    orbitals = []  # list to the nodes in the HDF5 containing the MOs
    energies = []
    guess_job = None

    # orbital type is either an empty string for restricted calculation
    # or alpha/beta for unrestricted calculations
    orbitals_type = config.orbitals_type

    for j, gs in enumerate(config.geometries):

        # number of the point with respect to all the trajectory
        k = j + config.enumerate_from

        # dictionary containing the information of the j-th job
        dict_input = defaultdict(lambda: None)  # type:  DefaultDict[str, Any]
        dict_input["geometry"] = gs
        dict_input["k"] = k

        # Path where the MOs will be store in the HDF5
        dict_input["node_MOs"] = [
            join(orbitals_type, "eigenvalues", f"point_{k}"),
            join(orbitals_type, "coefficients", f"point_{k}"),
            join(orbitals_type, "occupation", f"point_{k}"),
        ]

        dict_input["node_energy"] = join(orbitals_type, "energy", f"point_{k}")

        # If the MOs are already store in the HDF5 format return the path
        # to them and skip the calculation
        if config["compute_orbitals"]:
            predicate = dict_input["node_MOs"]
        else:
            predicate = dict_input["node_energy"]

        if is_data_in_hdf5(config.path_hdf5, predicate):
            logger.info(f"point_{k} has already been calculated")
            orbitals.append(dict_input["node_MOs"])
        else:
            logger.info(f"point_{k} has been scheduled")

            # Add cell parameters from file if given
            if file_cell_parameters is not None:
                adjust_cell_parameters(general, array_cell_parameters, j)
            # Path to I/O files
            dict_input["point_dir"] = config.folders[j]
            dict_input["job_files"] = create_file_names(
                dict_input["point_dir"], k)
            dict_input["job_name"] = f'point_{k}'

            # Compute the MOs and return a new guess
            promise_qm = compute_orbitals(config, dict_input, guess_job)

            # Check if the job finishes succesfully
            promise_qm = schedule_check(promise_qm, config, dict_input)

            # Store the computation
            if config["compute_orbitals"]:
                orbitals.append(store_molecular_orbitals(config, dict_input, promise_qm))
            else:
                orbitals.append(None)
            energies.append(store_enery(config, dict_input, promise_qm))

            guess_job = promise_qm

    return gather(gather(*orbitals), gather(*energies))


@schedule
def store_molecular_orbitals(
        config: DictConfig, dict_input: DefaultDict[str, Any], promise_qm: PromisedObject) -> str:
    """Store the MOs in the HDF5.

    Returns
    -------
    str
        Node path in the HDF5

    """
    # Molecular Orbitals
    mos = promise_qm.orbitals

    # Store in the HDF5
    try:
        save_orbitals_in_hdf5(mos, config, dict_input["job_name"])
    # Remove the ascii MO file
    finally:
        if config.remove_log_file:
            work_dir = promise_qm.archive['work_dir']
            path_mos = fnmatch.filter(os.listdir(work_dir), 'mo_*MOLog')[0]
            os.remove(join(work_dir, path_mos))

    return dict_input["node_MOs"]


def save_orbitals_in_hdf5(mos: OrbitalType, config: DictConfig, job_name: str) -> None:
    """Store the orbitals from restricted and unrestricted calculations."""
    if isinstance(mos, CP2KInfoMO):
        dump_orbitals_to_hdf5(mos, config, job_name)
    else:
        alphas, betas = mos
        dump_orbitals_to_hdf5(alphas, config, job_name, "alphas")
        dump_orbitals_to_hdf5(betas, config, job_name, "betas")


def dump_orbitals_to_hdf5(
    data: CP2KInfoMO,
    config: DictConfig,
    job_name: str,
    orbitals_type: str = "",
) -> None:
    """Store the result in HDF5 format.

    Parameters
    ----------
    data
        Tuple of energies and coefficients of the molecular orbitals
    config
        Dictionary with the job configuration
    job_name
        Name of the current job
    orbitals_type
        Either an empty string for MO coming from a restricted job or alpha/beta
        for unrestricted MO calculation
    """
    for name, array in zip(("eigenvalues", "coefficients"), (data.eigenvalues, data.eigenvectors)):
        path_property = join(orbitals_type, name, job_name)
        store_arrays_in_hdf5(config.path_hdf5, path_property, array)

    # Store the number of occupied and virtual orbitals as a size-2 dataset.
    # Occupied in this context is equivalent to "non-zero occupation"
    path_property = join(orbitals_type, "occupation", job_name)
    occ_array = np.array(data.get_nocc_nvirt(), dtype=np.int64)
    store_arrays_in_hdf5(config.path_hdf5, path_property, occ_array, dtype=np.int64)


@schedule
def store_enery(
        config: DictConfig, dict_input: Dict, promise_qm: PromisedObject) -> str:
    """Store the total energy in the HDF5 file.

    Returns
    -------
    str
        Node path to the energy in the HDF5

    """
    store_arrays_in_hdf5(
        config.path_hdf5, dict_input['node_energy'], promise_qm.energy)

    logger.info(
        f"Total energy of point {dict_input['k']} is: {promise_qm.energy}")

    return dict_input["node_energy"]


def compute_orbitals(
        config: DictConfig, dict_input: Dict, guess_job: PromisedObject) -> PromisedObject:
    """Call a Quantum chemisty package to compute the MOs.

    When finish store the MOs in the HdF5 and returns a new guess.
    """
    dict_input["job_files"] = create_file_names(
        dict_input["point_dir"], dict_input["k"])

    # Calculating initial guess
    compute_guess = config.calc_new_wf_guess_on_points is not None

    # A job  is a restart if guess_job is None and the list of
    # wf guesses are not empty
    is_restart = guess_job is None and compute_guess

    pred = (dict_input['k']
            in config.calc_new_wf_guess_on_points) or is_restart

    general = config.cp2k_general_settings

    if pred:
        guess_job = prepare_job_cp2k(
            general["cp2k_settings_guess"], dict_input, guess_job)

    promise_qm = prepare_job_cp2k(
        general["cp2k_settings_main"], dict_input, guess_job)

    return promise_qm


@schedule
def schedule_check(
        promise_qm: PromisedObject, config: DictConfig, dict_input: DictConfig) -> PromisedObject:
    """Check wether a calculation finishes succesfully otherwise run a new guess."""
    job_name = dict_input["job_name"]
    point_dir = dict_input["point_dir"]

    # Warnings of the computation
    warnings = promise_qm.warnings

    # Check for SCF convergence errors
    if not config.ignore_warnings and warnings is not None and any(
            w == SCF_Convergence_Warning for msg, w in warnings.items()):
        # Report the failure
        msg = f"Job: {job_name} Finished with Warnings: {warnings}"
        logger.warning(msg)

        # recompute a new guess
        msg1 = "Computing a new wave function guess for job: {job_name}"
        logger.warning(msg1)

        # Remove the previous ascii file containing the MOs
        msg2 = f"removing file containig the previous failed MOs of {job_name}"
        logger.warning(msg2)
        path = fnmatch.filter(os.listdir(point_dir), 'mo*MOLog')[0]
        os.remove(join(point_dir, path))

        # Compute new guess at point k
        config.calc_new_wf_guess_on_points.append(dict_input["k"])
        return compute_orbitals(config, dict_input, None)
    else:
        return promise_qm


def create_point_folder(
        work_dir: str | os.PathLike[str], n: int, enumerate_from: int) -> List[str]:
    """Create a new folder for each point in the MD trajectory."""
    folders = []
    for k in range(enumerate_from, n + enumerate_from):
        new_dir = join(work_dir, f'point_{k}')
        os.makedirs(new_dir, exist_ok=True)
        folders.append(new_dir)

    return folders


def split_file_geometries(path_xyz: PathLike) -> Sequence[str]:
    """Read a set of molecular geometries in xyz format."""
    # Read Cartesian Coordinates
    with open(path_xyz) as f:
        xss = iter(f.readlines())

    data = []
    while True:
        try:
            natoms = int(next(xss).split()[0])
            molecule = "".join([next(xss) for _ in range(natoms + 1)])
            data.append(f"{natoms}\n{molecule}")
        except StopIteration:
            break

    return data


def create_file_names(work_dir: str | os.PathLike[str], i: int) -> JobFiles:
    """Create a namedTuple with the name of the 4 files used for each point in the trajectory."""
    file_xyz = join(work_dir, f'coordinates_{i}.xyz')
    file_inp = join(work_dir, f'point_{i}.inp')
    file_out = join(work_dir, f'point_{i}.out')
    file_mo = join(work_dir, f'mo_coeff_{i}.out')

    return JobFiles(file_xyz, file_inp, file_out, file_mo)


def adjust_cell_parameters(
        general: DictConfig,
        array_cell_parameters: Matrix,
        j: int) -> None:
    """Adjust the cell parameters on the fly.

    If the cell parameters change during the MD simulations, adjust them
    for the molecular orbitals computation.
    """
    for s in (
        general[p] for p in (
            'cp2k_settings_main',
            'cp2k_settings_guess')):
        s.cell_parameters = array_cell_parameters[j, 2:11].reshape(
            3, 3).tolist()
        s.cell_angles = None
