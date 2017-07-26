__author__ = "Felipe Zapata"


# ================> Python Standard  and third-party <==========
from .initialization import read_time_dependent_coeffs
from collections import namedtuple
from multiprocessing import Pool
from nac.common import (
    Matrix, Tensor3D, Vector, change_mol_units, product, retrieve_hdf5_data)
from nac.schedule.components import calculate_mos
from noodles import gather
from os.path import join
from qmworks import run
from qmworks.parsers import parse_string_xyz
from typing import (Dict, List, Tuple)
import numpy as np
import logging

GridCube = namedtuple("GridCube", ("voxel", "shape"))

# Get logger
logger = logging.getLogger(__name__)


def compute_cubes(
        package_name: str, project_name: str, package_args: Dict,
        path_time_coeffs: str=None, grid_data: Tuple=None,
        guess_args: Dict=None, geometries: List=None,
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
    # Molecular orbital calculations calculatios
    mo_paths_hdf5 = calculate_mos(
        package_name, geometries, project_name, path_hdf5, traj_folders,
        package_args, guess_args, calc_new_wf_guess_on_points,
        enumerate_from, package_config=package_config,
        ignore_warnings=ignore_warnings)

    # geometries in atomic units
    molecules_au = [center_molecule_in_cube(center_molecule_in_cube(
        grid_data, change_mol_units(parse_string_xyz(gs))))
        for gs in geometries]

    # Time-dependent coefficients
    time_depend_coeffs = read_time_dependent_coeffs(path_time_coeffs)
    msg = "Reading time_dependent coefficients from: {}".format(path_time_coeffs)
    logger.info(msg)

    # Compute the values in the given grid
    promised_grids = [compute_grid_density(
        grid_data, path_hdf5, time_depend_coeffs, dictCGFs, mol, path_mo)
        for mol, path_mo in zip(molecules_au, mo_paths_hdf5)]

    # Execute the workflow
    path_grids = run(gather(*promised_grids), folder=work_dir)

    print_grids(path_hdf5, path_grids)


def compute_grid_density(
        grid_data: Tuple, path_hdf5: str, time_depend_coeffs: Matrix,
        dictCGFs: Dict, mol: List, path_mo: str) -> str:
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
    grid_coordinates = create_grid_nuclear_coordinates(grid_data, mol)

    # Compute the density using all the Avialable CPUs
    density_grid = distribute_grid_computation(mol, grid_coordinates, dictCGFs)

    # # Read the molecular orbitals from the HDF5
    # css = retrieve_hdf5_data(path_hdf5, path_mo)

    return density_grid


def create_grid_nuclear_coordinates(grid_data: Tuple, mol: List) -> Tensor3D:
    """
    Compute all the Nuclear coordinates where the density is evaluated
    """
    pass
    # size_grid = product(grid_data.shape)

    # grid = np.emtpty((len(mol), 3), size_grid)


def molecular_linspace(mol: List, points: int, delta: float):
    """
    Create an Array containing the grid in cartesian coordinate for all the atoms.
    """
    pass
    # coords = np.array([at.xyz for at in mol])
    # n_atoms = len(mol)

    # # Create The whole grid
    # grid = np.empty(points ** 3, n_atoms, 3)

    # xs = np.stack([nuclear_linspace(xyz, 0, points, delta) for at in mol)]

    

def nuclear_linspace(xyz: Vector, axis: int, points: int, delta: float) -> Matrix:
    """
    Create a Matrix containing the displacement for a single Nuclear coordinate
    """
    arr = np.empty((3, points))
    for k, x in enumerate(xyz):
        if k == axis:
            arr[k] = space_fun(x, points, delta)
        else:
            arr[k] = np.repeat(x, points)

    return arr.transpose()


def space_fun(center, points, delta):
    """
    Create a 1D array Grid for a component of a Nuclear coordinate
    """
    start = center - points * 0.5 * delta
    stop = center + points * 0.5 * delta
    return np.linspace(start, stop, points, endpoint=False) + 0.5 * delta


def distribute_grid_computation(
        mol: List, grid_coordinates: Tensor3D, dictCGFs: Tensor3D) -> Tensor3D:
    """
    Use all the available CPUs to compute the grid of density for a given
    molecular geometry.
    """
    pass
    # with Pool() as p:
    #     p.map = 


def print_grids(path_hdf5: str, path_grids: List) -> None:
    """
    Write the grid in plain text format
    """
    pass


def center_molecule_in_cube(grid_data: Tuple, mol: List) -> None:
    """
    Translate a molecule to the center of the cube
    """
    return mol
