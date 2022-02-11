"""Initial configuration setup.

Index
-----
.. currentmodule:: nanoqm.workflows.initialization
.. autosummary::
    initialize
    split_trajectory
API
---
.. autofunction:: initialize
.. autofunction:: split_trajectory

"""

from __future__ import annotations

__all__ = ['initialize', 'read_swaps', 'split_trajectory']

import fnmatch
import getpass
import logging
import os
import subprocess
import tempfile
from os.path import join
from pathlib import Path
from subprocess import PIPE, Popen
from typing import List, Union

import numpy as np
import pkg_resources
from qmflows.parsers import parse_string_xyz
from qmflows.parsers.cp2KParser import readCp2KBasis
from qmflows.type_hints import PathLike

from ..common import (BasisFormats, DictConfig, Matrix, change_mol_units,
                      is_data_in_hdf5, retrieve_hdf5_data,
                      store_arrays_in_hdf5)
from ..schedule.components import create_point_folder, split_file_geometries

# Starting logger
logger = logging.getLogger(__name__)


def initialize(config: DictConfig) -> DictConfig:
    """Initialize all the data required to schedule the workflows.

    Returns
    -------
    DictConfig
        Input to run the workflow.

    """
    log_config(config)

    # Scratch folder
    scratch_path = create_path_option(config["scratch_path"])
    if scratch_path is None:
        scratch_path = Path(tempfile.gettempdir()) / \
            getpass.getuser() / config.project_name
        logger.warning(
            f"path to scratch was not defined, using: {scratch_path}")
    config['workdir'] = scratch_path

    # If the directory does not exist create it
    if not scratch_path.exists():
        scratch_path.mkdir(parents=True)

    # Touch HDF5 if it doesn't exists
    if not os.path.exists(config.path_hdf5):
        Path(config.path_hdf5).touch()

    # all_geometries type :: [String]
    geometries = split_file_geometries(config["path_traj_xyz"])
    config['geometries'] = geometries

    # Create a folder for each point the the dynamics
    enumerate_from = config["enumerate_from"]
    len_geometries = len(geometries)
    config["folders"] = create_point_folder(
        scratch_path, len_geometries, enumerate_from)

    config['calc_new_wf_guess_on_points'] = guesses_to_compute(
        config['calculate_guesses'], enumerate_from, len_geometries)

    # Generate a list of tuples containing the atomic label
    # and the coordinates to generate the primitive CGFs
    atoms = parse_string_xyz(geometries[0])
    if 'angstrom' in config["geometry_units"].lower():
        atoms = change_mol_units(atoms)

    # Save Basis to HDF5
    save_basis_to_hdf5(config)

    return config


def save_basis_to_hdf5(config: DictConfig) -> None:
    """Store the specification of the basis set in the HDF5 to compute the integrals."""
    root: str = config['cp2k_general_settings']['path_basis']
    files: "None | list[str]" = config['cp2k_general_settings']['basis_file_name']
    if files is not None:
        basis_paths = [os.path.join(root, i) for i in files]
    else:
        basis_paths = [os.path.join(root, "BASIS_MOLOPT")]

    for path in basis_paths:
        if not is_data_in_hdf5(config.path_hdf5, path):
            store_cp2k_basis(config.path_hdf5, path)


def store_cp2k_basis(path_hdf5: PathLike, path_basis: PathLike) -> None:
    """Read the CP2K basis set into an HDF5 file."""
    keys, vals = readCp2KBasis(path_basis)
    node_paths_exponents = [
        join("cp2k/basis", xs.atom.lower(), xs.basis, "exponents") for xs in keys
    ]
    node_paths_coefficients = [
        join("cp2k/basis", xs.atom.lower(), xs.basis, "coefficients") for xs in keys
    ]

    exponents = [xs.exponents for xs in vals]
    coefficients = [xs.coefficients for xs in vals]
    formats = [str(xs.basisFormat) for xs in keys]

    store_arrays_in_hdf5(path_hdf5, node_paths_exponents, exponents)
    store_arrays_in_hdf5(path_hdf5, node_paths_coefficients, coefficients,
                         attribute=BasisFormats(name="basisFormat", value=formats))


def guesses_to_compute(calculate_guesses: str, enumerate_from: int, len_geometries: int) -> List[int]:
    """Guess for the wave function."""
    if calculate_guesses is None:
        points_guess = []
    elif calculate_guesses.lower() in 'first':
        # Calculate new Guess in the first geometry
        points_guess = [enumerate_from]
        msg = "An initial Calculation will be computed as guess for the wave function"
        logger.info(msg)
    elif calculate_guesses.lower() in 'all':
        # Calculate new Guess in each geometry
        points_guess = [enumerate_from + i for i in range(len_geometries)]
        msg = "A guess calculation will be done for each geometry"
        logger.info(msg)

    return points_guess


def read_swaps(path_hdf5: Union[str, Path], project_name: str) -> Matrix:
    """Read the crossing tracking for the Molecular orbital."""
    path_swaps = join(project_name, 'swaps')
    if is_data_in_hdf5(path_hdf5, path_swaps):
        return retrieve_hdf5_data(path_hdf5, path_swaps)
    else:
        msg = f"""There is not a tracking file called: {path_swaps}
        This file is automatically created when running the worflow_coupling
        simulations"""
        raise RuntimeError(msg)


def split_trajectory(path: str | Path, nblocks: int, pathOut: str | os.PathLike[str]) -> List[str]:
    """Split an XYZ trajectory in n Block and write them in a given path.

    Parameters
    ----------
    path
        Path to the XYZ file.
    nblocks
        number of Block into which the xyz file is split.
    pathOut
        Path were the block are written.

    Returns
    -------
    list
        list of paths to the xyz geometries

    """
    with open(path, 'r') as f:
        # Read First line
        ls = f.readline()
        numat = int(ls.split()[0])

    # Number of lines in the file
    cmd = f"wc -l {path}"
    ls = subprocess.check_output(cmd.split()).decode()
    lines = int(ls.split()[0])
    if (lines % (numat + 2)) != 0:
        lines += 1

    # Number of points in the xyz file
    npoints = lines // (numat + 2)
    # Number of points for each chunk
    nchunks = int(np.ceil(npoints / nblocks))
    # Number of lines per block
    lines_per_block = nchunks * (numat + 2)
    # Path where the splitted xyz files are written
    prefix = join(pathOut, 'chunk_xyz_')
    cmd = f'split -a 1 -l {lines_per_block} {path} {prefix}'
    output = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    rs = output.communicate()
    err = rs[1]
    if err:
        raise RuntimeError(f"Submission Errors: {err.decode()}")
    else:
        return fnmatch.filter(os.listdir(), "chunk_xyz_?")


def log_config(config: DictConfig) -> None:
    """Print initial configuration."""
    workdir = os.path.abspath('.')
    file_log = f'{config.project_name}.log'
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(asctime)s---%(levelname)s\n%(message)s\n',
                        datefmt='[%I:%M:%S]')
    logging.getLogger("noodles").setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.terminator = ""

    version = pkg_resources.get_distribution('nano-qmflows').version
    path = pkg_resources.resource_filename('nanoqm', '')

    logger.info(f"Using nano-qmflows version: {version} ")
    logger.info(f"nano-qmflows path is: {path}")
    logger.info(f"Working directory is: {workdir}")
    logger.info(f"Data will be stored in HDF5 file: {config.path_hdf5}")


def create_path_option(path: str | os.PathLike[str]) -> Path | None:
    """Create a Path object or return None if path is None."""
    return Path(path) if path is not None else None
