__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========

from nac.schedule.components import calculate_mos
from nac.common import (
    Matrix, Tensor3D, Vector, change_mol_units, femtosec2au,
    retrieve_hdf5_data)
from nac.schedule.scheduleET import compute_overlaps_ET
from nac.integrals.electronTransfer import photo_excitation_rate
from noodles import (gather, schedule)
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
        basis_name: str=None,
        enumerate_from: int=0, package_config: Dict=None,
        calc_new_wf_guess_on_points: str=None, guess_args: Dict=None,
        work_dir: str=None, traj_folders: List=None,
        dictCGFs: Dict=None, nHOMO: int=None, couplings_range: Tuple=None,
        pyxaid_range: Tuple=None, fragment_indices: None=List,
        dt: float=1):
    """
    Use a md trajectory to calculate the Electron transfer rate.

    :param package_name: Name of the package to run the QM simulations.
    :param project_name: Folder name where the computations
    are going to be stored.
    :param package_args: Specific settings for the package.
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param geometries: List of string cotaining the molecular geometries.
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :param calc_new_wf_guess_on_points: number of Computations that used a
                                        previous calculation as guess for the
                                        wave function.
    :param nHOMO: index of the HOMO orbital.
    :param couplings_range: Range of MO use to compute the nonadiabatic
    :param pyxaid_range: range of HOMOs and LUMOs used by pyxaid.
    :param enumerate_from: Number from where to start enumerating the folders
                           create for each point in the MD.
    :param traj_folders: List of paths to where the CP2K MOs are printed.
     :param package_config: Parameters required by the Package.
    :param fragment_indices: indices of atoms belonging to a fragment.
    :param dt: integration time used in the molecular dynamics.
    :returns: None
    """
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

    # Time-dependent coefficients
    time_depend_coeffs = read_time_dependent_coeffs(
        path_time_coeffs, pyxaid_range)

    # compute_overlaps_ET
    scheduled_overlaps = schedule(compute_overlaps_ET)
    fragment_overlaps = scheduled_overlaps(
        project_name, molecules_au, basis_name, path_hdf5, dictCGFs,
        mo_paths_hdf5, fragment_indices, enumerate_from, package_name)

    # Delta time in a.u.
    dt_au = dt * femtosec2au

    # Indices relation between the PYXAID active space and the orbitals
    # stored in the HDF5
    map_index_pyxaid_hdf5 = create_map_index_pyxaid(
        nHOMO, couplings_range, pyxaid_range)

    # Electron transfer rate for each frame of the Molecular dynamics
    etrs = [compute_photoexcitation(
        i, path_hdf5, molecules_au[i: i + 3], time_depend_coeffs[i: i + 3],
        fragment_overlaps[i: i + 3], map_index_pyxaid_hdf5, dt_au)
        for i in range(nPoints)]

    # Execute the workflow
    electronTransferRates = run(gather(*etrs))

    rs = list(map(lambda ts: '{:10.6f} {:10.6f}\n'.format(*ts),
                  electronTransferRates))
    result = ''.join(rs)

    with open("ElectronTranferRates", "w") as f:
        f.write(result)


def compute_photoexcitation(
        i: int, path_hdf5: str, geometries: List, time_dependent_coeffs: Matrix,
        fragment_overlaps: List, map_index_pyxaid_hdf5: Matrix,
        dt_au: float) -> List:
    """
    :param i: Electron transfer rate at time i * dt
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param geometries: list of 3 contiguous molecular geometries
    :param time_depend_paths: mean value of the time dependent coefficients
    computed with PYXAID.
    param fragment_overlaps: Tensor containing 3 overlap matrices corresponding
    with the `geometries`.
    :param map_index_pyxaid_hdf5: map from PYXAID excitation to the indices i,j
    of the molecular orbitals stored in the HDF5.
    :param dt_au: Delta time in atomic units
    :returns: promise to path to the Coupling inside the HDF5.
    """
    scheduled_photoexcitation = schedule(photo_excitation_rate)
    overlaps = np.stack(retrieve_hdf5_data(path_hdf5, fragment_overlaps))

    return scheduled_photoexcitation(
        geometries, overlaps, time_dependent_coeffs, map_index_pyxaid_hdf5,
        dt_au)


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
        path_pyxaid_out: str, pyxaid_range: Tuple) -> Tensor3D:
    """
    :param path_pyxaid_out: Path to the out of the NA-MD carried out by
    PYXAID.
    :type path_pyxaid_out: String
    :param pyxaid_range: range of HOMOs and LUMOs used by pyxaid.
    :returns: Numpy array
    """
    # Read output files
    files_out = os.listdir(path_pyxaid_out)
    names_out_pop  = fnmatch.filter(files_out, "out*")
    paths_out_pop = (join(path_pyxaid_out, x) for x in names_out_pop)

    # Read the data
    pss = map(parse_population, paths_out_pop)

    return np.mean(np.stack(pss), axis=1)


def create_map_index_pyxaid(
        homo: int, couplings_range: Tuple, pyxaid_range: Tuple) -> Matrix:
    """
    Creating an index mapping from PYXAID to the content of the HDF5.
    """
    # Check user-defined PYXAID activate space and shift indices to 0
    pyxaid_range = pyxaid_range if pyxaid_range is not None else couplings_range
    pyxaid_homos, pyxaid_lumos = pyxaid_range

    # Number of HOMOS and LUMOS defined for PYXAID (counting from 0)
    pyxaid_LUMO = pyxaid_homos

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
        indexes_hdf5[i] = compute_excitation_indexes(i)

    return indexes_hdf5


def check_larger_than_zero(val: int) -> int:
    """
    Check than an index is larger than zero
    """
    if val >= 0:
        return val
    else:
        msg = "The molecular orbitals is out of range, check the `pyxaid_range` var"
        raise RuntimeError(msg)
