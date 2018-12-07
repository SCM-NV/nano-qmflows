__all__ = ['create_map_index_pyxaid', 'initialize', 'read_swaps',
           'split_trajectory', 'store_transf_matrix']

from nac.basisSet import (compute_normalization_sphericals, create_dict_CGFs)
from nac.common import (
    Matrix, Tensor3D, Vector, change_mol_units, retrieve_hdf5_data, search_data_in_hdf5)
from nac.integrals import calc_transf_matrix
from nac.schedule.components import (
    create_point_folder, split_file_geometries)
from os.path import join
from qmflows.hdf5.quantumHDF5 import StoreasHDF5
from qmflows.parsers import parse_string_xyz
from subprocess import (PIPE, Popen)
from typing import (Dict, List, Tuple)

import fnmatch
import getpass
import h5py
import logging
import nac
import numpy as np
import os
import pkg_resources
import subprocess


# Starting logger
logger = logging.getLogger(__name__)


def initialize(
        project_name: str=None, path_traj_xyz: str=None, basis_name: str=None,
        enumerate_from: int=0, calculate_guesses: int='first',
        path_hdf5: str=None, scratch_path: str=None, path_basis: str=None,
        path_potential: str=None, geometry_units: str='angstrom', **kwargs) -> Dict:
    """
    Initialize all the data required to schedule the workflows associated with
    the nonadaibatic coupling
    """
    # Start logging event
    file_log = '{}.log'.format(project_name)
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(levelname)s:%(message)s  %(asctime)s\n',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    # User variables
    username = getpass.getuser()

    # Scratch
    if scratch_path is None:
        scratch_path = join('/tmp', username, project_name)
        logger.warning("path to scratch was not defined, using: {}".format(scratch_path))

    # If the directory does not exist create it
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # Cp2k configuration files
    cp2k_config = {"basis": path_basis, "potential": path_potential}

    # HDF5 path
    if path_hdf5 is None:
        path_hdf5 = join(scratch_path, 'quantum.hdf5')
        logger.warning("path to the HDF5 was not defined, using: {}".format(path_hdf5))

    # all_geometries type :: [String]
    geometries = split_file_geometries(path_traj_xyz)

    # Create a folder for each point the the dynamics
    traj_folders = create_point_folder(scratch_path, len(geometries),
                                       enumerate_from)
    if calculate_guesses is None:
        points_guess = []
    elif calculate_guesses.lower() in 'first':
        # Calculate new Guess in the first geometry
        points_guess = [enumerate_from]
        msg = "An initial Calculation will be computed as guess for the wave function"
        logger.info(msg)
    else:
        # Calculate new Guess in each geometry
        points_guess = [enumerate_from + i for i in range(len(geometries))]
        msg = "A guess calculation will be done for each geometry"
        logger.info(msg)

    # Generate a list of tuples containing the atomic label
    # and the coordinates to generate
    # the primitive CGFs
    atoms = parse_string_xyz(geometries[0])
    if 'angstrom' in geometry_units.lower():
        atoms = change_mol_units(atoms)

    # CGFs per element
    dictCGFs = create_dict_CGFs(path_hdf5, basis_name, atoms,
                                package_config=cp2k_config)

    # Calculcate the matrix to transform from cartesian to spherical
    # representation of the overlap matrix
    hdf5_trans_mtx = store_transf_matrix(
        path_hdf5, atoms, dictCGFs, basis_name, project_name)

    d = {'package_config': cp2k_config, 'path_hdf5': path_hdf5,
         'calc_new_wf_guess_on_points': points_guess,
         'geometries': geometries, 'enumerate_from': enumerate_from,
         'dictCGFs': dictCGFs, 'work_dir': scratch_path,
         'folders': traj_folders, 'basis_name': basis_name,
         'hdf5_trans_mtx': hdf5_trans_mtx, "nHOMO": kwargs["nHOMO"],
         "mo_index_range": kwargs["mo_index_range"]}

    return d


def read_time_dependent_coeffs(
        path_pyxaid_out: str) -> Tensor3D:
    """
    :param path_pyxaid_out: Path to the out of the NA-MD carried out by
    PYXAID.
    :returns: Numpy array
    """
    # Read output files
    files_out = os.listdir(path_pyxaid_out)
    names_out_pop = fnmatch.filter(files_out, "out*")
    paths_out_pop = (join(path_pyxaid_out, x) for x in names_out_pop)

    # Read the data
    pss = map(parse_population, paths_out_pop)

    # Returns 3D-Array containing the TD-coefficients
    return np.array(list(pss))


def parse_population(filePath: str) -> Matrix:
    """
    returns a matrix contaning the pop for each time in each row.
    """
    with open(filePath, 'r') as f:
        xss = f.readlines()
    rss = [[float(x) for i, x in enumerate(l.split())
            if i % 2 == 1 and i > 2] for l in xss]

    return np.array(rss)


def create_map_index_pyxaid(
        orbitals_range: Tuple, pyxaid_HOMO: int, pyxaid_Nmin: int,
        pyxaid_Nmax: int) -> Matrix:
    """
    Creating an index mapping from PYXAID to the content of the HDF5.
    """
    number_of_HOMOs = pyxaid_HOMO - pyxaid_Nmin + 1
    number_of_LUMOs = pyxaid_Nmax - pyxaid_HOMO

    # Shift range to start counting from 0
    pyxaid_Nmax -= 1
    pyxaid_Nmin -= 1

    # Pyxaid LUMO counting from 0
    pyxaid_LUMO = pyxaid_HOMO

    def compute_excitation_indexes(index_ext: int) -> Vector:
        """
        create the index of the orbitals involved in the excitation i -> j.
        """
        # final state
        j_index = pyxaid_LUMO + (index_ext // number_of_HOMOs)
        # initial state
        i_index = pyxaid_Nmin + (index_ext % number_of_HOMOs)

        return np.array((i_index, j_index), dtype=np.int32)

    # Generate all the excitation indexes of pyxaid including the ground state
    number_of_indices = number_of_HOMOs * number_of_LUMOs
    indexes_hdf5 = np.empty((number_of_indices + 1, 2), dtype=np.int32)

    # Ground state
    indexes_hdf5[0] = pyxaid_Nmin, pyxaid_Nmin

    for i in range(number_of_indices):
        indexes_hdf5[i + 1] = compute_excitation_indexes(i)

    return indexes_hdf5


def read_swaps(path_hdf5: str, project_name: str) -> Matrix:
    """
    Read the crossing tracking for the Molecular orbital
    """
    path_swaps = join(project_name, 'swaps')
    if search_data_in_hdf5(path_hdf5, path_swaps):
        return retrieve_hdf5_data(path_hdf5, path_swaps)
    else:
        msg = """There is not a tracking file called: {}
        This file is automatically created when running the worflow_coupling
        simulations""".format(path_swaps)
        raise RuntimeError(msg)


def store_transf_matrix(
        path_hdf5: str, atoms: List, dictCGFs: Dict, basis_name: str,
        project_name: str, package_name: str='cp2k') -> str:
    """
    calculate the transformation of the overlap matrix from both spherical
    to cartesian and from cartesian to spherical.

    :param path_hdf5: Path to the HDF5 file.
    :param atoms: Atoms that made up the molecule.
    :param project_name: Name of the project.
    :param package_name: Name of the ab initio simulation package.
    :returns: Numpy matrix containing the transformation matrix.
    """
    # Norms of the spherical CGFs for each element
    dict_global_norms = compute_normalization_sphericals(dictCGFs)
    # Compute the transformation matrix between cartesian and spherical
    path = os.path.join(project_name, 'trans_mtx')
    with h5py.File(path_hdf5) as f5:
        if path not in f5:
            mtx = calc_transf_matrix(
                f5, atoms, basis_name, dict_global_norms, package_name)
            store = StoreasHDF5(f5, package_name)
            store.funHDF5(path, mtx)
    return path


def split_trajectory(path: str, nBlocks: int, pathOut: str) -> List:
    """
    Split an XYZ trajectory in n Block and write
    them in a given path.
    :Param path: Path to the XYZ file.
    :param nBlocks: number of Block into which the xyz file is split.
    :param pathOut: Path were the block are written.
    :returns: path to block List
    """
    with open(path, 'r') as f:
        # Read First line
        ls = f.readline()
        numat = int(ls.split()[0])

    # Number of lines in the file
    cmd = "wc -l {}".format(path)
    ls = subprocess.check_output(cmd.split()).decode()
    lines = int(ls.split()[0])
    # Number of points in the xyz file
    nPoints = lines // (numat + 2)
    # Number of points for each chunk
    nChunks = nPoints // nBlocks
    # Number of lines per block
    lines_per_block = nChunks * (numat + 2)
    # Path where the splitted xyz files are written
    prefix = join(pathOut, 'chunk_xyz_')
    cmd = 'split -a 1 -l {} {} {}'.format(lines_per_block, path, prefix)
    subprocess.run(cmd, shell=True)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    rs = p.communicate()
    err = rs[1]
    if err:
        raise RuntimeError("Submission Errors: {}".format(err))
    else:
        return fnmatch.filter(os.listdir(), "chunk_xyz*")


def log_config(work_dir, path_hdf5, algorithm):
    """
    Print initial configuration
    """
    # Get logger
    logger = logging.getLogger(__name__)

    version = pkg_resources.get_distribution('qmflows-namd').version
    path = nac.__path__

    logger.info("Using qmflows-namd version: {} ".format(version))
    logger.info("qmflows-namd path is: {}".format(path))
    logger.info("Working directory is: {}".format(work_dir))
    logger.info("Data will be stored in HDF5 file: {}".format(path_hdf5))
    logger.info("The chosen algorithm to compute the coupling is: {}\n".format(algorithm))
