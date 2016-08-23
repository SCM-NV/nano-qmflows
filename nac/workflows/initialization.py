
__all__ = ['store_transf_matrix', 'initialize']

from os.path import join
from nac.common import change_mol_units
from nac.integrals import calc_transf_matrix
from nac.schedule.components import (create_dict_CGFs, create_point_folder,
                                     split_file_geometries)
from qmworks.hdf5.quantumHDF5 import StoreasHDF5
from qmworks.parsers import parse_string_xyz

import getpass
import h5py
import os
import shutil
# ====================================<>=======================================


def initialize(project_name, path_traj_xyz, basisname, enumerate_from=0,
               calculate_guesses='first',
               scratch="/scratch-shared", path_basis=None, path_potential=None,
               dt=1, geometry_units='angstrom'):
    """
    Initialize all the data required to schedule the workflows associated with
    the nonadaibatic coupling
    """
    # User variables
    cwd = os.path.realpath(".")
    username = getpass.getuser()

    # Scratch
    scratch = "/scratch-shared"
    scratch_path = join(scratch, username, project_name)

    # Create Work_dir if it does not exist
    work_dir = os.path.join(cwd, project_name)
    # remove previous
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)

    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # Cp2k configuration files
    cp2k_config = {"basis": path_basis, "potential": path_potential}

    # HDF5 path
    path_hdf5 = join(scratch_path, 'quantum.hdf5')

    # all_geometries type :: [String]
    geometries = split_file_geometries(path_traj_xyz)

    # Create a folder for each point the the dynamics
    traj_folders = create_point_folder(work_dir, len(geometries),
                                       enumerate_from)

    if calculate_guesses.lower() in 'first':
        # Calculate new Guess in the first geometry
        points_guess = [enumerate_from]
    else:
        # Calculate new Guess in each geometry
        points_guess = [enumerate_from + i for i in range(len(geometries))]

    # Generate a list of tuples containing the atomic label
    # and the coordinates to generate
    # the primitive CGFs
    atoms = parse_string_xyz(geometries[0])
    if 'angstrom' in geometry_units.lower():
        atoms = change_mol_units(atoms)

    dictCGFs = create_dict_CGFs(path_hdf5, basisname, atoms, cp2k_config)

    # Calculcate the matrix to transform from cartesian to spherical
    # representation of the overlap matrix
    hdf5_trans_mtx = store_transf_matrix(path_hdf5, atoms, basisname,
                                         project_name, packageName='cp2k')

    d = {'package_config': cp2k_config, 'path_hdf5': path_hdf5,
         'calc_new_wf_guess_on_points': points_guess,
         'geometries': geometries, 'enumerate_from': enumerate_from,
         'dt': 1, 'dictCGFs': dictCGFs, 'work_dir': work_dir,
         'traj_folders': traj_folders, 'basisname': basisname,
         'hdf5_trans_mtx': hdf5_trans_mtx, 'first_geometry': atoms}

    return d


def store_transf_matrix(path_hdf5, atoms, basisName, project_name,
                        packageName):
    """
    calculate the transformation of the overlap matrix from both spherical
    to cartesian and from cartesian to spherical.

    :param path_hdf5: Path to the HDF5 file.
    :type: String
    :param atoms: Atoms that made up the molecule.
    :type atoms:  List of Strings
    :param project_name: Name of the project.
    :type project_name: String
    :param packageName: Name of the ab initio simulation package.
    :type packageName: String
    :returns: Numpy matrix containing the transformation matrix.
    """
    with h5py.File(path_hdf5) as f5:
        mtx = calc_transf_matrix(f5, atoms, basisName, packageName)
        store = StoreasHDF5(f5, packageName)
        path = os.path.join(project_name, 'trans_mtx')
        store.funHDF5(path, mtx)
    return path
