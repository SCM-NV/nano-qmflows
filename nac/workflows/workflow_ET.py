__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from os.path import join

import fnmatch
import h5py
import numpy as np
import os
import plams

# ==============================> Internal modules <===========================
from .components import calculate_mos
from nac.common import (change_mol_units, retrieve_hdf5_data)
from nac.integrals.electronTransfer import photoExcitationRate
from nac.workflows import initialize
from noodles import gather
from qmworks import run
from qmworks.parsers import parse_string_xyz

# ==============================> Main <==================================


def search_data_in_hdf5(path_hdf5, path_to_prop):
    """
    Search if the node exists in the HDF5 file.
    """
    with h5py.File(path_hdf5, 'r') as f5:
        if isinstance(path_to_prop, list):
            pred = all(path in f5 for path in path_to_prop)
        else:
            pred = path_to_prop in f5

    return pred


def calculate_ETR(package_name, project_name, package_args, geometries=None,
                  pathTimeCoeffs=None, initial_conditions=None,
                  path_hdf5=None, enumerate_from=0, package_config=None,
                  calc_new_wf_guess_on_points=None, guess_args=None,
                  work_dir=None, traj_folders=None, basisname=None,
                  dictCGFs=None, hdf5_trans_mtx=None):
    """
    Use a md trajectory to calculate the Electron transfer rate
    nmad.

    :param package_name: Name of the package to run the QM simulations.
    :type  package_name: String
    :param project_name: Folder name where the computations
    are going to be stored.
    :type project_name: String
    :param package_args: Specific settings for the package
    :type package_args: dict
    :param geometries: List of string cotaining the molecular geometries
                       numerical results.
    :type path_traj_xyz: [String]
    :param calc_new_wf_guess_on_points: number of Computations that used a
                                        previous calculation as guess for the
                                        wave function.
    :type calc_new_wf_guess_on_points: Int
    :param enumerate_from: Number from where to start enumerating the folders
                           create for each point in the MD.
    :type enumerate_from: Int
    :param package_config: Parameters required by the Package.
    :type package_config: Dict

    :returns: None
    """
    # Initial conditions
    if initial_conditions is None:
        initial_conditions = [0]

    # Time-dependent coefficients
    time_depend_coeffs = retrieve_hdf5_data(path_hdf5, pathTimeCoeffs)

    # prepare Cp2k Job
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(package_name, geometries, project_name,
                                  path_hdf5, traj_folders, package_args,
                                  guess_args, calc_new_wf_guess_on_points,
                                  enumerate_from,
                                  package_config=package_config)

    # Number of ETR points calculated with the MD trajectory
    nPoints = len(geometries) - 2

    # List of tuples containing the electron transfer rates
    if hdf5_trans_mtx is not None:
            trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx)

    etrs = [schedule_photoexcitation(i, path_hdf5, dictCGFs, geometries,
                                     time_depend_coeffs, mo_paths_hdf5,
                                     trans_mtx=trans_mtx)
            for i in range(nPoints)]

    electronTransferRates = run(gather(*etrs))

    rs = list(map(lambda ts: '{:10.6f} {:10.6f}\n'.format(*ts),
                  electronTransferRates))
    result = ''.join(rs)

    with open("ElectronTranferRates", "w") as f:
        f.write(result)

# ==============================> Tasks <======================================


def schedule_photoexcitation(i, path_hdf5, dictCGFs, all_geometries,
                             time_depend_paths, mo_paths, trans_mtx=None,
                             enumerate_from=0, units='angstrom'):
    """
    :param i: nth coupling calculation.
    :type i: Int
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :type path_hdf5: String
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :param all_geometries: list of molecular geometries
    :type all_geometries: String list
    :param time_depend_paths: Path to the time-dependent coefficients
    calculated with PYXAID and stored in HDF5 format.
    :type time_depend_paths: [String]
    :param mo_paths: Paths to the MO coefficients and energies in the
    HDF5 file.
    :type mo_paths: [String]
    :param trans_mtx: transformation matrix from cartesian to spherical
    orbitals.
    :type trans_mtx: Numpy Array
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :type enumerate_from: Int
    :returns: promise to path to the Coupling inside the HDF5

    """
    j, k = i + 1, i + 2

    xss = all_geometries[i], all_geometries[j], all_geometries[k]
    geometries = tuple(map(parse_string_xyz, xss))
    if 'angstrom' in units.lower():
        geometries = tuple(map(change_mol_units, geometries))

    mos = tuple(map(lambda j:
                    retrieve_hdf5_data(path_hdf5,
                                       mo_paths[i + j][1]), range(3)))
    time_coeffs = tuple(map(lambda j:
                            retrieve_hdf5_data(path_hdf5,
                                               time_depend_paths[i + j]),
                            range(3)))

    return photoExcitationRate(geometries, dictCGFs, time_coeffs, mos,
                               trans_mtx=trans_mtx)


def parse_population(filePath):
    """
    returns a matrix contaning the pop for each time in each row.
    """
    with open(filePath, 'r') as f:
        xss = f.readlines()
    rss = [[float(x) for i, x in enumerate(l.split())
            if i % 2 == 1 and i > 2] for l in xss]

    return np.array(rss)


def read_time_dependent_coeffs(path_hdf5, pathProperty, path_pyxaid_out):
    """
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :type path_hdf5: String
    :param pathProperty: path to the node that contains the time
    coeffficients.
    :type pathProperty: String
    :param path_pyxaid_out: Path to the out of the NA-MD carried out by
    PYXAID.
    :type path_pyxaid_out: String
    :returns: None
    """
    # Read output files
    files_out = os.listdir(path_pyxaid_out)
    names_out_es, names_out_pop  = [fnmatch.filter(files_out, x) for x
                                    in ["*energies*", "out*"]]
    _, paths_out_pop = [[join(path_pyxaid_out, x) for x in xs]
                        for xs in [names_out_es, names_out_pop]]

    # ess = map(parse_energies, paths_out_es)
    pss = map(parse_population, paths_out_pop)

    # Make a 3D stack of arrays the calculate the mean value
    # for the same time
    # average_es = np.mean(np.stack(ess), axis=0)
    # average_pop = np.mean(np.stack(pss), axis=0)
    data = np.stack(pss)

    # Save Data in the HDF5 file
    with h5py.File(path_hdf5) as f5:
        f5.require_dataset(pathProperty, shape=np.shape(data),
                           data=data, dtype=np.float32)

# ================================> Main <=====================================


def main():
    plams.init()

    # Project
    project_name = 'NAC'
    path_traj_xyz = "./data/traj_3_points.xyz"
    pyxaid_out_dir = "./step3/out"
    basisname = "DZVP-MOLOPT-SR-GTH"
    home = os.path.expanduser('~')
    basiscp2k = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potcp2k = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")

    initial_config = initialize(project_name, path_traj_xyz, basisname,
                                path_basis=basiscp2k, path_potential=potcp2k,
                                enumerate_from=0)
    path_hdf5 = initial_config['path_hdf5']

    # Process PYXAID results and store them in HDF5
    pathProperty = join(project_name, "pyxaid/timeCoeffs")
    if search_data_in_hdf5(path_hdf5, pathProperty):
        read_time_dependent_coeffs(path_hdf5, pathProperty, pyxaid_out_dir)

    # Electron Transfer rate calculation
    pyxaid_initial_cond = [0, 24, 49]

    # Electron transfer rate computation computation
    cp2k_args = None
    calculate_ETR('cp2k', project_name, cp2k_args,
                  pathTimeCoeffs=pathProperty,
                  initial_conditions=pyxaid_initial_cond,
                  **initial_config)
    plams.finish()

# ==============<>=============
if __name__ == "__main__":
    main()
