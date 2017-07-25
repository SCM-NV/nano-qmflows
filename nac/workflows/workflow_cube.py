__author__ = "Felipe Zapata"


# ================> Python Standard  and third-party <==========
from collections import namedtuple
from nac.common import (
    Matrix, Tensor3D, Vector, change_mol_units, femtosec2au,
    retrieve_hdf5_data, search_data_in_hdf5)
from nac.schedule.components import calculate_mos
from noodles import gather
from os.path import join
from qmworks import run
from qmworks.parsers import parse_string_xyz
from typing import (Dict, List, Tuple)
import logging

GridCube = namedtuple("GridCube", ("origin", "voxels", "grid"))


def compute_cubes(
        package_name: str, project_name: str, package_args: Dict,
        grid_data: Tuple=None, guess_args: Dict=None, geometries: List=None,
        dictCGFs: Dict=None, calc_new_wf_guess_on_points: str=None,
        path_hdf5: str=None, enumerate_from: int=0, package_config: Dict=None,
        traj_folders: List=None, work_dir: str=None, basisname: str=None,
        hdf5_trans_mtx: str=None, nHOMO: int=None, algorithm='levine',
        ignore_warnings=False) -> None:
    """
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param geometries: List of string cotaining the molecular geometries.
    """
    # Start logging event
    file_log = '{}.log'.format(project_name)
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(levelname)s:%(message)s  %(asctime)s\n',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    # Molecular orbital calculations calculatios
    mo_paths_hdf5 = calculate_mos(
        package_name, geometries, project_name, path_hdf5, traj_folders,
        package_args, guess_args, calc_new_wf_guess_on_points,
        enumerate_from, package_config=package_config,
        ignore_warnings=ignore_warnings)

    # geometries in atomic units
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for gs in geometries]

    promised_grids = [compute_grid_density(
        grid_data, path_hdf5, dictCGFs, mol, path_mo)
        for mol, path_mo in zip(molecules_au, mo_paths_hdf5)]

    path_grids = run(gather(*promised_grids), folder=work_dir)

    print_grids(path_hdf5, path_grids)


def compute_grid_density(
        grid_data: Tuple, path_hdf5: str, dictCGFs: Dict, mol: List,
        path_mo: str) -> str:
    """
    Compute the grid density for a given geometry and store it in the HDF5

    :param grid_data: Tuple containing the grid specification
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :param mol: List of atomic labels and cartesian coordinates
    :param path_mo: Path to the MO in the HDF5.
    :returns: path where the grid is stored in the HDF5
    """
    pass


def print_grids(path_hdf5 :str, path_grids: List) -> None:
    """
    Write the grid in plain text format
    """
    pass
