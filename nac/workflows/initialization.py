__all__ = ['initialize', 'split_trajectory', 'store_transf_matrix']

from os.path import join
from nac.basisSet import (compute_normalization_sphericals, create_dict_CGFs)
from nac.common import change_mol_units
from nac.integrals import calc_transf_matrix
from nac.schedule.components import (
    create_point_folder, split_file_geometries)
from qmworks.hdf5.quantumHDF5 import StoreasHDF5
from qmworks.parsers import parse_string_xyz
from subprocess import (PIPE, Popen)
from typing import (Dict, List)

import fnmatch
import getpass
import h5py
import logging
import os
import subprocess

# Starting logger
logger = logging.getLogger(__name__)
# ====================================<>=======================================


def initialize(
        project_name: str, path_traj_xyz: str, basisname: str,
        enumerate_from: int=0, calculate_guesses: int='first',
        path_hdf5: str=None, scratch_path: str=None, path_basis: str=None,
        path_potential: str=None, geometry_units: str='angstrom') -> Dict:
    """
    Initialize all the data required to schedule the workflows associated with
    the nonadaibatic coupling
    """
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
    dictCGFs = create_dict_CGFs(path_hdf5, basisname, atoms,
                                package_config=cp2k_config)

    # Calculcate the matrix to transform from cartesian to spherical
    # representation of the overlap matrix
    hdf5_trans_mtx = store_transf_matrix(
        path_hdf5, atoms, dictCGFs, basisname, project_name)

    d = {'package_config': cp2k_config, 'path_hdf5': path_hdf5,
         'calc_new_wf_guess_on_points': points_guess,
         'geometries': geometries, 'enumerate_from': enumerate_from,
         'dictCGFs': dictCGFs, 'work_dir': scratch_path,
         'traj_folders': traj_folders, 'basisname': basisname,
         'hdf5_trans_mtx': hdf5_trans_mtx}

    return d


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
            l = f.readline()  # Read First line
            numat = int(l.split()[0])

    # Number of lines in the file
    cmd = "wc -l {}".format(path)
    l = subprocess.check_output(cmd.split()).decode()
    lines = int(l.split()[0])
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
