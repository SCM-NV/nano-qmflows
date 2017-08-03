__author__ = "Felipe Zapata"

__all__ = ["workflow_compute_cubes"]

# ================> Python Standard  and third-party <==========
from .initialization import (
    create_map_index_pyxaid, read_time_dependent_coeffs)
from collections import namedtuple
from functools import partial
from itertools import repeat
from multiprocessing import (cpu_count, Pool)
from nac.common import (
    Array, AtomXYZ, Matrix, Tensor3D, Vector, change_mol_units,
    getmass, retrieve_hdf5_data, store_arrays_in_hdf5)
from nac.integrals.multipoleIntegrals import calcOrbType_Components
from nac.integrals.nonAdiabaticCoupling import compute_range_orbitals
from nac.schedule.components import calculate_mos
from noodles import (gather, schedule)
from os.path import join
from qmworks import run
from qmworks.parsers import parse_string_xyz
from scipy import sparse
from typing import (Dict, List, Tuple)
import numpy as np
import logging

# Data types
GridCube = namedtuple("GridCube", ("voxel", "shape"))
CGFS = namedtuple("CGFS", ("primitives", "ang_expo"))

# Get logger
logger = logging.getLogger(__name__)


def workflow_compute_cubes(
        package_name: str, project_name: str, package_args: Dict,
        path_time_coeffs: str=None, grid_data: Tuple=None,
        guess_args: Dict=None, geometries: List=None,
        dictCGFs: Dict=None, calc_new_wf_guess_on_points: str=None,
        path_hdf5: str=None, enumerate_from: int=0, package_config: Dict=None,
        traj_folders: List=None, work_dir: str=None, basisname: str=None,
        hdf5_trans_mtx: str=None, nHOMO: int=None, orbitals_range: Tuple=None,
        pyxaid_HOMO: int=None, pyxaid_Nmin: int=None, pyxaid_Nmax: int=None,
        time_steps_grid: int=1, ignore_warnings=False, **kwargs) -> None:
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
    molecules_au = [center_molecule_in_cube(grid_data, center_molecule_in_cube(
        grid_data, change_mol_units(parse_string_xyz(gs))))
        for gs in geometries]

    # Extract the information of the CGFs dictionary
    dictCGFs_array = {l: CGFS(
        get_primitives(dictCGFs, l), get_angular_exponents(dictCGFs, l))
        for l in dictCGFs.keys()}

    # Nuclear coordinates of the grid
    grid_coordinates = create_grid_nuclear_coordinates(grid_data)

    np.save('grid_coordinates', grid_coordinates)

    # Retrieve the matrix to transform from Cartesian to spherical coordinates
    trans_mtx = sparse.csr_matrix(
        retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx))

    # Compute the values in the given grid
    promised_fun_grid = schedule(compute_grid_density)
    promised_grids = gather(*[promised_fun_grid(
        k, mol, project_name, grid_data, grid_coordinates, path_hdf5,
        dictCGFs_array, trans_mtx, mo_paths_hdf5, nHOMO, orbitals_range)
        for k, mol in enumerate(molecules_au)])

    # Compute the density weighted by the population computed with PYXAID
    # Time-dependent coefficients
    if path_time_coeffs is not None:
        time_depend_coeffs = read_time_dependent_coeffs(path_time_coeffs)
        msg = "Reading time_dependent coefficients from: {}".format(path_time_coeffs)
        logger.info(msg)
    else:
        time_depend_coeffs = None

    scheduled_density = schedule(compute_TD_density)
    promised_TD_density = scheduled_density(
        path_hdf5, promised_grids, time_depend_coeffs, orbitals_range,
        pyxaid_HOMO, pyxaid_Nmax, pyxaid_Nmin, time_steps_grid)

    # Execute the workflow
    path_grids, grids_TD = run(
        gather(promised_grids, promised_TD_density), folder=work_dir)

    print_grids(
        grid_data, molecules_au[0],
        retrieve_hdf5_data(path_hdf5, path_grids[0])[:, 6], 'test.cube', 1)

    # Print the cube files
    if grids_TD is not None:
        for k, (grid, mol) in enumerate(zip(grids_TD, molecules_au)):
            step = time_steps_grid * k
            file_name = join(project_name, 'point_{}'.format(step))

            print_grids(grid_data, mol, grid, file_name, step)

    return path_grids


def compute_TD_density(
        path_hdf5: str, promised_grids: List, time_depend_coeffs: Matrix,
        orbitals_range: Tuple, pyxaid_HOMO: int, pyxaid_Nmin: int,
        pyxaid_Nmax: int, time_steps_grid: int):
    """
    Multiply the population with the corresponding orbitals
    """
    if all(x is not None for x in
           [orbitals_range, pyxaid_HOMO, pyxaid_Nmax, pyxaid_Nmin]):
        # Create a map from PYXAID orbitals to orbitals store in the HDF5
        map_indices = create_map_index_pyxaid(
            orbitals_range, pyxaid_HOMO, pyxaid_Nmin, pyxaid_Nmax)
        electron_indices = map_indices[:, 1]

        # Extract only the orbitals involved in the electron transfer
        for k, path_grid in enumerate(promised_grids):
            grid = retrieve_hdf5_data(path_hdf5, path_grid)
            electron_grid = grid[:, electron_indices]

            # Compute the time_dependent ET
            population = time_depend_coeffs[k * time_steps_grid]
            grid_ET = np.dot(electron_grid, population)

            yield grid_ET
    else:
        return None


def compute_grid_density(
        k: int, mol: List, project_name: str, grid_data: Tuple,
        grid_coordinates: Array,
        path_hdf5: str, dictCGFs_array: Dict, trans_mtx: Matrix,
        paths_mos: List, nHOMO: int, orbitals_range: Tuple) -> str:
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

    # Compute the values of the orbital using all the Avialable CPUs
    # Before multiplying for the MO coefficient
    orbital_grid_cartesian = distribute_grid_computation(
        mol, grid_coordinates, dictCGFs_array)

    # Transform to spherical coordinate
    transpose = trans_mtx.transpose()
    orbital_grid_sphericals = sparse.csr_matrix.dot(
        orbital_grid_cartesian, transpose)

    # # |phi| ^ 2
    # orbital_grid_sphericals *= orbital_grid_sphericals

    # Read the molecular orbitals from the HDF5
    css = retrieve_hdf5_data(path_hdf5, paths_mos[k][1])

    # Extract a subset of MOs from the HDF5
    lowest, highest = compute_range_orbitals(css, nHOMO, orbitals_range)
    css = css[:, lowest: highest]

    # # Ci ^ 2
    # css *= css

    # Ci^2 * |phi| ^ 2. Matrix shape: points ** 3, number_of_orbitals
    density_grid = np.dot(orbital_grid_sphericals, css)

    # Store the density grid in the HDF5
    store_arrays_in_hdf5(path_hdf5, path_grid, density_grid)

    return path_grid


def distribute_grid_computation(
        molecule: List, grid_coordinates: Matrix,
        dictCGFs_array: Tensor3D) -> Matrix:
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
    number_of_CGFs = sum(
        dictCGFs_array[at.symbol].primitives.shape[0] for at in molecule)

    # Distribute the jobs among the available CPUs
    with Pool() as p:
        grid = np.concatenate(
            p.map(partial(
                compute_CGFs_chunk, molecule, dictCGFs_array, number_of_CGFs),
                chunks))

    return grid


def compute_CGFs_chunk(
        molecule: List, dictCGFs_array: Dict, number_of_CGFs: int,
        chunk: Matrix) -> Matrix:
    """
    Compute the value of all the CGFs for a set molecular geometries
    taken as an slice of the grid.
    """
    # Dimensions of the result array
    chunk_size = chunk.shape[0]

    # Resulting array
    cgfs_grid = np.empty((chunk_size, number_of_CGFs))

    for i, voxel_center in enumerate(np.rollaxis(chunk, axis=0)):
        cgfs_grid[i] = compute_CGFs_values(
            molecule, dictCGFs_array, voxel_center, number_of_CGFs)
    return cgfs_grid


def compute_CGFs_values(
        molecule: List, dictCGFs_array: Dict, voxel_center: Vector,
        number_of_CGFs) -> Vector:
    """
    Evaluate the CGFs in a given molecular geometry.
    """
    vs = np.empty(number_of_CGFs)

    acc = 0
    for k, at in enumerate(molecule):
        cgfs = dictCGFs_array[at.symbol]
        size = cgfs.primitives.shape[0]
        deltaR = (at.xyz - voxel_center).reshape(1, 3)

        vs[acc: acc + size] = compute_CGFs_per_atom(cgfs, deltaR)
        acc += size

    return vs


def compute_CGFs_per_atom(cgfs: Tuple, deltaR: Vector) -> Vector:
    """
    Compute the value of the CGFs for a particular atom
    """
    ang_exponents = cgfs.ang_expo
    primitives = cgfs.primitives

    # Iterate over each CGF per atom
    rs = np.empty(primitives.shape[0])
    for k, (expos, ps) in enumerate(zip(ang_exponents, primitives)):
        x = compute_CGF(deltaR, expos, ps)
        rs[k] = x

    return rs


def compute_CGF(
        xyz: Vector, ang_expos: Vector, primitives: Matrix) -> float:
    """
    Compute a single CGF
    """
    coeffs = primitives[0]
    expos = primitives[1].reshape(primitives[1].size, 1)

    # Compute the xyz gaussian primitives
    xs = xyz ** ang_expos
    gaussians = xs * np.exp(-expos * xyz ** 2)

    # Multiplied the x,y and z gaussians
    rs_gauss = np.prod(gaussians, axis=1)

    # multiple the gaussian by the contraction coefficients
    return np.sum(coeffs * rs_gauss)


def create_grid_nuclear_coordinates(grid_data: Tuple) -> Matrix:
    """
    Compute all the Nuclear coordinates where the density is evaluated

    :returns: 4D-Array containing the voxels center
    """
    shape = grid_data.shape
    voxel = grid_data.voxel

    # Vector of equally seperated voxels in 1D
    xs = np.linspace(0, voxel * shape, num=shape, endpoint=False)

    # Create 4D Grid containing the voxel centers
    grids = np.stack(np.meshgrid(xs, xs, xs), axis=3)

    return grids.reshape(shape ** 3, 3)


def get_primitives(dictCGFs: Dict, s: str) -> Tensor3D:
    """ Extract the primitives of the CGFs for an atom as a 3D tensor"""
    return np.stack([cgf.primitives for cgf in dictCGFs[s]])


def get_angular_exponents(dictCGFs: Dict, s: str) -> List:
    """ Extract the CGFs angular labels for an atom """
    ls = [cgf.orbType for cgf in dictCGFs[s]]

    return np.array(
        [[calcOrbType_Components(l, x) for x in range(3)] for l in ls])


def center_molecule_in_cube(grid_data: Tuple, mol: List) -> None:
    """
    Translate a molecule to the center of the cube
    """
    return [AtomXYZ(at.symbol, np.array(at.xyz)) for at in mol]


def print_grids(
        grid_data: Tuple, molecule: List, grid: Vector,
        file_name: str, step: int) -> None:
    """
    Write the grid cube format format
    """
    fmt1 = '{:5d}{:12.6f}{:12.6f}{:12.6f}\n'

    # Box specification
    shape = grid_data.shape
    voxel = grid_data.voxel

    # First two lines
    header1 = "Cube file generated by qmworks-namd\n"
    header2 = "Contains time depedent grid step: {}\n".format(step)

    # Box matrix
    numat = fmt1.format(len(molecule), 0, 0, 0)
    vectorx = fmt1.format(shape, voxel, 0, 0)
    vectory = fmt1.format(shape, 0, voxel, 0)
    vectorz = fmt1.format(shape, 0, 0, voxel)

    # Print coordinates in the cube format
    coords = format_coords(molecule)

    # print the grid in the cube format
    arr = format_cube_grid(shape, grid)

    s = header1 + header2 + numat + vectorx + vectory + vectorz + coords + arr

    with open(file_name, 'w') as f:
        f.write(s)


def format_coords(mol: List) -> str:
    """ Format the nuclear coordinates in the cube style"""
    fmt = '{:5d}{:12.6f}{:12.6f}{:12.6f}{:12.6f}\n'
    s = ''
    for at in mol:
        s += fmt.format(getmass(at.symbol), 0, *at.xyz)

    return s


def format_cube_grid(shape: int, grid: Vector) -> str:
    """ Format the numerical grid in the cube style"""

    arr = grid.reshape(shape ** 2, shape)

    # remaining format
    r = shape % 6
    if r != 0:
        fmt_r = ''.join(repeat(' {:12.5E}', r)) + '\n'
    else:
        fmt_r = None

    # number of row per chunk
    nrow = shape // 6

    # cube are printing in lines of maximum 6 columns
    s = ''
    for chunk in np.rollaxis(arr, axis=0):
        s += format_chunk_grid(chunk, nrow, r, fmt_r)

    return s


def format_chunk_grid(
        chunk: Vector, nrow: int, r: int, fmt_r: str) -> str:
    """
    format a vector containing an slice of the grid in the cube format.
    """
    # 6 column cube format
    fmt = ' {:12.5E} {:12.5E} {:12.5E} {:12.5E} {:12.5E} {:12.5E}\n'

    # Reshape the vector in a matrix with 6 columns
    head = chunk[:nrow * 6].reshape(nrow, 6)

    # Print Columns of 6 numbers
    s = ''
    for v in head:
        s += fmt.format(*v)

    # Print the remaining values
    if r != 0:
        s += fmt_r.format(*chunk[-r:])

    return s
