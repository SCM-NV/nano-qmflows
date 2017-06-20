__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========

from .components import calculate_mos
from nac.common import (Matrix, Tensor3D, Vector, change_mol_units,
                        retrieve_hdf5_data)
from nac.schedule.scheduleET import compute_overlaps_ET
from nac.integrals.electronTransfer import photoExcitationRate
from noodles import gather
from os.path import join
from qmworks import run
from qmworks.parsers import parse_string_xyz

import fnmatch
import numpy as np
import os

from typing import (Dict, List, Tuple)

# ==============================> Main <==================================


def calculate_ETR(
        package_name: str, project_name: str, package_args: Dict,
        path_time_coeffs: str=None, geometries: List=None,
        initial_conditions: List=None, path_hdf5: str=None,
        enumerate_from: int=0, package_config: Dict=None,
        calc_new_wf_guess_on_points: str=None, guess_args: Dict=None,
        work_dir: str=None, traj_folders: List=None,
        dictCGFs: Dict=None, hdf5_trans_mtx: str=None,
        nHOMO: int=None, couplings_range: Tuple=None,
        pyxaid_homo: int=None, pyxaid_range: Tuple=None,
        fragment_indices: None=List):
    """
    Use a md trajectory to calculate the Electron transfer rate
    nmad.

    :param package_name: Name of the package to run the QM simulations.
    :param project_name: Folder name where the computations
    are going to be stored.
    :param package_args: Specific settings for the package
    :param geometries: List of string cotaining the molecular geometries
                       numerical results.
    :type path_traj_xyz: [String]
    :param calc_new_wf_guess_on_points: number of Computations that used a
                                        previous calculation as guess for the
                                        wave function.
    :param enumerate_from: Number from where to start enumerating the folders
                           create for each point in the MD.
    :param package_config: Parameters required by the Package.

    :returns: None
    """
    map_index_pyxaid_hdf5 = create_map_index_pyxaid(
        nHOMO, couplings_range, pyxaid_range)

    # Time-dependent coefficients
    time_depend_coeffs = read_time_dependent_coeffs(path_hdf5, path_time_coeffs)

    # prepare Cp2k Job
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(
        package_name, geometries, project_name, path_hdf5, traj_folders,
        package_args, guess_args, calc_new_wf_guess_on_points,
        enumerate_from, package_config=package_config)

    # Number of ETR points calculated with the MD trajectory
    nPoints = len(geometries) - 2

    # geometries in atomic units
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for gs in geometries]

    # compute_overlaps_ET
    fragment_overlaps = compute_overlaps_ET(
        project_name, molecules_au, path_hdf5, mo_paths_hdf5, hdf5_trans_mtx,
        fragment_indices, dictCGFs, enumerate_from)

    etrs = [compute_photoexcitation(
        i, path_hdf5, molecules_au[i: i + 3], time_depend_coeffs[i: i + 3],
        fragment_overlaps[i: i + 3], map_index_pyxaid_hdf5, enumerate_from)
        for i in range(nPoints)]

    # Execute the workflow
    electronTransferRates = run(gather(*etrs))

    rs = list(map(lambda ts: '{:10.6f} {:10.6f}\n'.format(*ts),
                  electronTransferRates))
    result = ''.join(rs)

    with open("ElectronTranferRates", "w") as f:
        f.write(result)


def compute_photoexcitation(
        i: int, path_hdf5: str, geometries: List, time_depend_coeffs: Tensor3D,
        fragment_overlaps, map_index_pyxaid_hdf5, enumerate_from: int,
        units: str='angstrom') -> List:
    """
    :param i: nth coupling calculation.
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :type path_hdf5: String
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :param geometries: list of 3 molecular geometries
    :param time_depend_paths: Path to the time-dependent coefficients
    calculated with PYXAID and stored in HDF5 format.
    :param mo_paths: Paths to the MO coefficients and energies in the
    HDF5 file.
    :param trans_mtx: transformation matrix from cartesian to spherical
    orbitals.
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :returns: promise to path to the Coupling inside the HDF5.
    """
    overlaps = np.stack(retrieve_hdf5_data(path_hdf5,))
    
    return photoExcitationRate(geometries, dictCGFs, time_depend_coeffs, mos,
                               trans_mtx=trans_mtx)


def parse_population(filePath: str) -> Matrix:
    """
    returns a matrix contaning the pop for each time in each row.
    """
    with open(filePath, 'r') as f:
        xss = f.readlines()
    rss = [[float(x) for i, x in enumerate(l.split())
            if i % 2 == 1 and i > 2] for l in xss]

    return np.array(rss)


def read_time_dependent_coeffs(
        path_hdf5: str, path_pyxaid_out: str) -> Tensor3D:
    """
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param path_pyxaid_out: Path to the out of the NA-MD carried out by
    PYXAID.
    :type path_pyxaid_out: String
    :returns: None
    """
    # Read output files
    files_out = os.listdir(path_pyxaid_out)
    names_out_pop  = fnmatch.filter(files_out, "out*")
    paths_out_pop = (join(path_pyxaid_out, x) for x in names_out_pop)

    # Read the data
    pss = map(parse_population, paths_out_pop)

    return np.stack(pss)


def create_map_index_pyxaid(
        homo: int, couplings_range: Tuple, pyxaid_HOMO: int,
        pyxaid_range: Tuple) -> Matrix:
    """
    Creating an index mapping from PYXAID to the content of the HDF5.
    """
    # Check user-defined PYXAID activate space
    pyxaid_range = pyxaid_range if pyxaid_range is not None else couplings_range
    pyxaid_homos, pyxaid_lumos = pyxaid_range

    # Number of HOMOS and LUMOS defined for PYXAID
    pyxaid_LUMO = pyxaid_homos + 1

    # index of the lowest orbital in the HDF5 used by PYXAID
    lowest_homo = check_larger_than_zero(couplings_range[0] - pyxaid_homos)

    def compute_excitation_indexes(index_ext: int) -> Vector:
        """
        create the index of the orbitals involved in the excitation i -> j.
        """
        # final state
        j_index = lowest_homo + pyxaid_LUMO + (index_ext // pyxaid_homos)
        # initial state
        i_index = lowest_homo + (index_ext % pyxaid_homos)

        return np.array((i_index, j_index), dtype=np.int32)

    # Generate all the excitation indexes of pyxaid
    number_of_indices = pyxaid_homos * pyxaid_lumos

    indexes_hdf5 = np.empty((number_of_indices, 2), dtype=np.int32)

    for i in range(number_of_indices):
        indexes_hdf5[i] = compute_excitation_indexes[i]


def check_larger_than_zero(val: int) -> int:
    """
    Check than an index is larger than zero
    """
    if val >= 0:
        return val
    else:
        msg = "The molecular orbitals is out of range, check the `pyxaid_range` var"
        raise RuntimeError(msg)
