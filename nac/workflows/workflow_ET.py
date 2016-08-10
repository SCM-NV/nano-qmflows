__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from os.path import join

import fnmatch
import getpass
import h5py
import numpy as np
import os
import plams
import shutil

# ==============================> Internal modules <===========================
from .components import (calculate_mos, create_dict_CGFs, create_point_folder,
                         split_file_geometries)
from nac.common import (change_mol_units, retrieve_hdf5_data)
from nac.schedule.scheduleCoupling import schedule_transf_matrix
from nac.integrals.electronTransfer import photoExcitationRate
from noodles import gather
from qmworks import run
from qmworks.parsers import parse_string_xyz
from qmworks.utils import flatten

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


def calculate_ETR(package_name, project_name, all_geometries, cp2k_args,
                  pathTimeCoeffs=None, initial_conditions=[0],
                  path_hdf5=None, enumerate_from=0, package_config=None,
                  calc_new_wf_guess_on_points=[0], guess_args=None):
    """
    Use a md trajectory to calculate the Electron transfer rate
    nmad.

    :param package_name: Name of the package to run the QM simulations.
    :type  package_name: String
    :param project_name: Folder name where the computations
    are going to be stored.
    :type project_name: String
    :param all_geometries: List of string cotaining the molecular geometries
    numerical results.
    :type path_traj_xyz: [String]
    :param package_args: Specific settings for the package
    :type package_args: dict
    :param use_wf_guess_each: number of Computations that used a previous
    calculation as guess for the wave function.
    :type use_wf_guess_each: Int
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :type enumerate_from: Int
    :param package_config: Parameters required by the Package.
    :type package_config: Dict
    :returns: None
    """
    # Create Work_dir if it does not exist
    cwd = os.path.realpath(".")
    work_dir = os.path.join(cwd, project_name)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)

    # Create Work_dir if it does not exist
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # Generate a list of tuples containing the atomic label
    # and the coordinates to generate
    # the primitive CGFs
    basisName = cp2k_args.basis
    atoms = parse_string_xyz(all_geometries[0])
    dictCGFs = create_dict_CGFs(path_hdf5, basisName, atoms, package_config)

    # Calculcate the matrix to transform from cartesian to spherical
    # representation of the overlap matrix
    hdf5_trans_mtx = schedule_transf_matrix(path_hdf5, atoms,
                                            basisName, project_name,
                                            packageName=package_name)
    
    # Create a folder for each point the the dynamics
    traj_folders = create_point_folder(work_dir, len(all_geometries),
                                       enumerate_from)

    # Time-dependent coefficients
    time_depend_coeffs = retrieve_hdf5_data(path_hdf5, pathTimeCoeffs)
    
    # prepare Cp2k Job
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(package_name, all_geometries, project_name,
                                  path_hdf5, traj_folders, cp2k_args,
                                  guess_args, calc_new_wf_guess_on_points,
                                  enumerate_from, package_config=package_config)

    # Number of ETR points calculated with the MD trajectory
    nPoints = len(all_geometries) - 2

    # List of tuples containing the electron transfer rates
    if hdf5_trans_mtx is not None:
            trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx)

    etrs = [schedule_photoexcitation(i, path_hdf5, dictCGFs, all_geometries,
                                     time_depend_coeffs, mo_paths_hdf5,
                                     trans_mtx=trans_mtx)
            for i in range(nPoints)]

    electronTransferRates = run(gather(*etrs))

    result = flatten(map(lambda ts: '{:10.6f} {:10.6f}\n'.format(*ts),
                         electronTransferRates))

    with open("ElectronTranferRates", "w") as f:
        f.write(result)

# ==============================> Tasks <=======================================


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
    paths_out_es, paths_out_pop = [[join(path_pyxaid_out, x) for x in xs]
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
    
# ==============================> Main <==================================


def main():
    plams.init()

    # User variables
    username = getpass.getuser()

    # Project
    project_name = 'NAC'

    # Path to the MD geometries
    path_traj_xyz = "./data/traj_3_points.xyz"

    # Work_dir
    scratch = "/scratch-shared"
    scratch_path = join(scratch, username, project_name)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # HDF5 path
    path_hdf5 = join(scratch_path, 'quantum.hdf5')

    # PYXAID Results
    pyxaid_out_dir = "./step3/out"

    # Process PYXAID results and store them in HDF5
    pathProperty = join(project_name, "pyxaid/timeCoeffs")
    if search_data_in_hdf5(path_hdf5, pathProperty):
        read_time_dependent_coeffs(path_hdf5, pathProperty, pyxaid_out_dir)

    # Named the points of the MD starting from this number
    enumerate_from = 0

    # all_geometries type :: [String]
    geometries = split_file_geometries(path_traj_xyz)

    # Electron Transfer rate calculation
    pyxaid_initial_cond = [0, 24, 49]
    
    # Electron transfer rate computation computation
    cp2k_args = None
    calculate_ETR('cp2k', project_name, geometries, cp2k_args,
                  pathTimeCoeffs=pathProperty,
                  initial_conditions=pyxaid_initial_cond,
                  path_hdf5=path_hdf5, enumerate_from=enumerate_from)

    print("PATH TO HDF5:{}\n".format(path_hdf5))
    plams.finish()

# ==============<>=============
    
if __name__ == "__main__":
    main()
