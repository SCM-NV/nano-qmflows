__author__ = "Felipe Zapata"


# ================> Python Standard  and third-party <==========
# from .initialization import read_time_dependent_coeffs
from collections import namedtuple
from functools import partial
from multiprocessing import (cpu_count, Pool)
from nac.common import (
    Array, Matrix, Tensor3D, Vector, change_mol_units, retrieve_hdf5_data,
    store_arrays_in_hdf5)
from nac.integrals.multipoleIntegrals import calcOrbType_Components
from nac.schedule.components import calculate_mos
from noodles import (gather)
from os.path import join
from qmworks import run
from qmworks.parsers import parse_string_xyz
from typing import (Dict, List, Tuple)
import numpy as np
import logging

# Data types
GridCube = namedtuple("GridCube", ("voxel", "shape"))
CGFS = namedtuple("CGFS", ("primitives", "ang_expo"))

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
        ignore_warnings=False, **kwargs) -> None:
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

    # Extract the information of the CGFs dictionary
    dictCGFs_array = {l: CGFS(
        get_primitives(dictCGFs, l), get_angular_exponents(dictCGFs, l))
        for l in dictCGFs.keys()}

    # Compute the values in the given grid
    promised_grids = [compute_grid_density(
        k, project_name, grid_data, path_hdf5, dictCGFs_array, mol, path_mo)
        for k, mol, path_mo in enumerate(zip(molecules_au, mo_paths_hdf5))]

    # # Compute the density weighted by the population computed with PYXAID
    # # Time-dependent coefficients
    # time_depend_coeffs = read_time_dependent_coeffs(path_time_coeffs)
    # msg = "Reading time_dependent coefficients from: {}".format(path_time_coeffs)
    # logger.info(msg)

    # scheduled_density = schedule(compute_dynamic_density)
    # promised_dynamic_density = scheduled_density(
    #     path_hdf5, promised_grids, time_depend_coeffs)

    # Execute the workflow
    path_grids = run(gather(*promised_grids), folder=work_dir)

    print_grids(path_hdf5, path_grids)


def compute_dynamic_density(
        path_hdf5: str, promised_grids: List, time_depend_coeffs: Matrix):
    pass


def compute_grid_density(
        k: int, project_name: str, grid_data: Tuple, path_hdf5: str,
        time_depend_coeffs: Matrix, dictCGFs_array: Dict, mol: List,
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
    path_grid = join(project_name, 'density_grid_{}'.format(k))
    # Nuclear coordinates of the grid
    grid_coordinates = create_grid_nuclear_coordinates(grid_data, mol)

    # Atomic symbols
    symbols = [at.symbol for at in mol]

    # Compute the values of the orbital using all the Avialable CPUs
    # Before multiplying for the MO coefficient
    orbital_grid = distribute_grid_computation(
        symbols, grid_coordinates, dictCGFs_array)

    # |phi| ^ 2
    orbital_grid *= orbital_grid

    # Read the molecular orbitals from the HDF5
    css = retrieve_hdf5_data(path_hdf5, path_mo)

    # Ci ^ 2
    css *= css

    # Ci^2 * |phi| ^ 2. Matrix shape: points ** 3, number_of_orbitals
    density_grid = np.dot(orbital_grid, css)

    # Store the density grid in the HDF5
    store_arrays_in_hdf5(path_hdf5, path_grid, density_grid)

    return path_grid


def distribute_grid_computation(
        symbols: List, grid_coordinates: Tensor3D,
        dictCGFs_array: Tensor3D) -> Tensor3D:
    """
    Use all the available CPUs to compute the grid of density for a given
    molecular geometry.
    """
    # Available CPUs
    nCPUs = cpu_count()

    # Number of entries to compute for each CPU
    n_points = grid_coordinates.shape[0]
    chunk = n_points // nCPUs

    # Remaining entries
    rest = n_points % nCPUs

    # Compute the indices of the grid calculate by each cpu
    indices = []
    acc = 0
    for i in range(nCPUs):
        b = 1 if i < rest else 0
        upper = acc + chunk + b
        indices.append((acc, upper))
        acc = upper

    # Chunks of th grid to compute
    chunks = (grid_coordinates[lower: upper] for lower, upper in indices)

    # Number of CGFs in total
    number_of_CGFs = sum(dictCGFs_array[l].primitives.shape[0] for l in symbols)

    # Distribute the jobs among the available CPUs
    with Pool() as p:
        grid = np.concatenate(
            p.map(partial(
                compute_CGFs_chunk, symbols, dictCGFs_array, number_of_CGFs),
                chunks))

    return grid


def compute_CGFs_chunk(
        symbols: List, dictCGFs_array: Dict, chunk: Tensor3D,
        number_of_CGFs) -> Matrix:
    """
    Compute the value of all the CGFs for a set molecular geometries
    taken as an slice of the grid.
    """
    # Dimensions of the result array
    chunk_size = chunk.shape[0]

    # Resulting array
    cgfs_grid = np.empty((chunk_size, number_of_CGFs))

    for i, mtx_coord in np.rollaxis(chunk, axis=0):
        cgfs_grid = compute_CGFs_values(
            symbols, dictCGFs_array, mtx_coord, number_of_CGFs)

    return cgfs_grid


def compute_CGFs_values(
        symbols: List, dictCGFs_array: Dict, mtx_coord: Matrix,
        number_of_CGFs) -> Vector:
    """
    Evaluate the CGFs in a given molecular geometry.
    """
    return np.concatenate([compute_CGFs_per_atom(dictCGFs_array, xyz)
                           for s, xyz in zip(symbols, mtx_coord)])


def compute_CGFs_per_atom(cgfs: Tuple, xyz: Vector) -> Vector:
    """
    Compute the value of the CGFs for a particular atom
    """
    ang_exponents = cgfs.ang_expo
    primitives = cgfs

    rs = np.array(primitives.shape[0])
    for k, (expos, ps) in enumerate(zip(ang_exponents, primitives)):
        rs[k] = compute_CGF(xyz, expos, ps)

    return rs


def compute_CGF(
        xyz: Vector, ang_expos: Vector, primitives: Matrix) -> float:
    """
    Compute a single CGF
    """
    return np.sum(np.prod(np.apply_along_axis(
        lambda t: gaussian_primitive(t[0], t[1], *primitives), 0,
        np.stack((ang_expos, xyz))), axis=1))


def gaussian_primitive(
        ang: float, x: float, coeff: Vector, expo: Vector) -> Vector:
    """Evaluate a primitive Gauss function"""
    return x ** ang * coeff * np.exp(-expo * x ** 2)


def create_grid_nuclear_coordinates(grid_data: Tuple, mol: List) -> Tensor3D:
    """
    Compute all the Nuclear coordinates where the density is evaluated
    """
    shape = grid_data.shape
    voxel = grid_data.voxel

    grids = np.stack([nuclear_linspace(
        at.xyz, shape, voxel) for at in mol], axis=3)

    return grids.reshape(shape ** 3, len(mol), 3)


def nuclear_linspace(xyz: Vector, points: int, delta: float) -> Array:
    """
    Create a Matrix containing the displacement for a single atomic coordinate
    """
    arr = np.empty((3, points))
    for k, x in enumerate(xyz):
        arr[k] = space_fun(x, points, delta)

    return np.stack(np.meshgrid(*arr, indexing='ij'), axis=3)


def space_fun(center, points, delta):
    """
    Create a 1D array Grid for a component of a Nuclear coordinate
    """
    start = center - points * 0.5 * delta
    stop = center + points * 0.5 * delta
    return np.linspace(start, stop, points, endpoint=False) + 0.5 * delta


def get_primitives(dictCGFs: Dict, s: str) -> Tensor3D:
    """ Extract the primitives of the CGFs for an atom as a 3D tensor"""
    return np.stack([cgf.primitives for cgf in dictCGFs[s]])


def get_angular_exponents(dictCGFs: Dict, s: str) -> List:
    """ Extract the CGFs angular labels for an atom """
    ls = [cgf.orbType for cgf in dictCGFs[s]]

    return np.array(
        [[calcOrbType_Components(l, x) for x in range(3)] for l in ls])


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
