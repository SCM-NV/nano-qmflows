__author__ = "Felipe Zapata"

__all__ = ["calculate_mos", "create_dict_CGFs", "create_point_folder",
           "split_file_geometries"]

# ================> Python Standard  and third-party <==========
from collections import namedtuple
from noodles import (gather, schedule)
from os.path import join

import fnmatch
import h5py
import logging
import os

# ==================> Internal modules <==========
from nac.basisSet.basisNormalization import createNormalizedCGFs
from nac.schedule.scheduleCp2k import prepare_job_cp2k
from qmworks.common import InputKey
from qmworks.hdf5 import dump_to_hdf5
from qmworks.hdf5.quantumHDF5 import (cp2k2hdf5, turbomole2hdf5)
from qmworks.utils import chunksOf
from qmworks.warnings_qmworks import SCF_Convergence_Warning

# Type Hints
from typing import (Dict, List, Tuple)

# ==============================<>=========================
# Tuple contanining file paths
JobFiles = namedtuple("JobFiles", ("get_xyz", "get_inp", "get_out", "get_MO"))

# Starting logger
logger = logging.getLogger(__name__)

# ==============================> Tasks <=====================================

def calculate_mos(package_name, all_geometries, project_name, path_hdf5,
                  folders, package_args, guess_args=None,
                  calc_new_wf_guess_on_points=None, enumerate_from=0,
                  package_config=None):
    """
    Look for the MO in the HDF5 file if they do not exists calculate them by
    splitting the jobs in batches given by the ``restart_chunk`` variables.
    Only the first job is calculated from scratch while the rest of the
    batch uses as guess the wave function of the first calculation in
    the batch.

    :param all_geometries: list of molecular geometries
    :type all_geometries: String list
    :param project_name: Name of the project used as root path for storing
    data in HDF5.
    :type project_name: String
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :type path_hdf5: String
    :param folders: path to the directories containing the MO outputs
    :type folders: String list
    :param package_args: Settings for the job to run.
    :type package_args: Settings
    :param calc_new_wf_guess_on_points: Calculate a new Wave function guess in
    each of the geometries indicated. By Default only an initial guess is
    computed.
    :type calc_new_wf_guess_on_points: [Int]
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :type enumerate_from: Int
    :returns: path to nodes in the HDF5 file to MO energies
              and MO coefficients.
    """
    def search_data_in_hdf5(xs):
        """
        Search if the node exists in the HDF5 file.
        """
        if os.path.exists(path_hdf5):
            with h5py.File(path_hdf5, 'r') as f5:
                if isinstance(xs, list):
                    return all(path in f5 for path in xs)
                else:
                    return xs in f5
        else:
            return False

    # First calculation has no initial guess
    guess_job = None

    # calculate the rest of the job using the previous point as initial guess
    orbitals = []  # list to the nodes in the HDF5 containing the MOs
    for j, gs in enumerate(all_geometries):

        # number of the point with respect to all the trajectory
        k = j + enumerate_from

        # Path where the MOs will be store in the HDF5
        root = join(project_name, 'point_{}'.format(k), package_name, 'mo')
        hdf5_orb_path = [join(root, 'eigenvalues'), join(root, 'coefficients')]

        # If the MOs are already store in the HDF5 format return the path
        # to them and skip the calculation
        if search_data_in_hdf5(hdf5_orb_path):
            logger.info("point_{} has already been calculated".format(k))
            orbitals.append(hdf5_orb_path)
        else:
            logger.info("Computing Molecular orbitals of: point_{}".format(k))

            # Path to I/O files
            point_dir = folders[j]
            job_files = create_file_names(point_dir, k)
            job_name = 'point_{}'.format(k)
            
            # Compute the MOs and return a new guess
            promise_qm = compute_orbitals(
                guess_job, package_name, project_name, path_hdf5,
                package_args, guess_args, package_config,
                calc_new_wf_guess_on_points, point_dir, job_files, k, gs)

            # Check if the job finishes succesfully
            promise_qm = schedule_check(
                promise_qm, job_name, package_name, project_name, path_hdf5,
                package_args, guess_args, package_config, point_dir, job_files, k, gs)
            
            # Store the computation
            path_MOs = store_in_hdf5(project_name, path_hdf5, promise_qm,
                                     hdf5_orb_path, job_name)

            # accumulate the MOs
            guess_job = promise_qm
            orbitals.append(path_MOs)

    return gather(*orbitals)


@schedule
def store_in_hdf5(project_name: str, path_hdf5: str, promise_qm: Tuple,
                  node_paths: str, job_name: str) -> None:
    #Molecular Orbitals
    mos = promise_qm.orbitals
    if mos is not None:
        with h5py.File(path_hdf5, 'r+') as f5:
            dump_to_hdf5(
                mos, 'cp2k', f5, project_name=project_name, job_name=job_name)
    return node_paths
            

def compute_orbitals(
        guess_job, package_name: str, project_name: str, path_hdf5: str,
        package_args: Dict, guess_args: Dict, package_config: Dict,
        calc_new_wf_guess_on_points: List, point_dir: str, job_files: Tuple,
        k: int, gs: List):
    """
    Call a Quantum chemisty package to compute the MOs required to calculate
    the nonadiabatic coupling. When finish store the MOs in the HdF5 and
    returns a new guess.
    """
    prepare_and_schedule = {'cp2k': prepare_job_cp2k}

    call_schedule_qm = prepare_and_schedule[package_name]

    # Calculating initial guess
    compute_guess = calc_new_wf_guess_on_points is not None

    # A job  is a restart if guess_job is None and the list of
    # wf guesses are not empty
    is_restart = guess_job is None and compute_guess

    pred = (k in calc_new_wf_guess_on_points) or is_restart
    
    if pred:
        print("Calling guess")
        guess_job = call_schedule_qm(
            gs, job_files, guess_args,
            k, point_dir, wfn_restart_job=guess_job,
            package_config=package_config)
        
    promise_qm = call_schedule_qm(
            gs, job_files, package_args,
            k, point_dir, wfn_restart_job=guess_job,
            package_config=package_config)
    
    return promise_qm

@schedule
def schedule_check(promise_qm, job_name: str, package_name: str,
                   project_name: str, path_hdf5: str, package_args: Dict,
                   guess_args: Dict, package_config: Dict, point_dir: str,
                   job_files: Tuple, k: int, gs: List):
    """
    Check wether a calculation finishes succesfully otherwise run a new guess.
    """
    # Warnings of the computation
    warnings = promise_qm.warnings

    # Check for SCF convergence errors
    if warnings is not None and any(
            w == SCF_Convergence_Warning for msg, w in warnings.items()):
        # Report the failure
        msg = "Job: {} Finished with Warnings: {}".format(job_name, warnings)
        logger.warning(msg)

        # recompute a new guess
        msg1 = "Computing a new wave function guess for job: {}".format(job_name)
        logger.warning(msg1)

        # Remove the previous ascii file containing the MOs 
        msg2 = "removing file containig the previous failed MOs of {}".format(job_name)
        logger.warning(msg2)
        path = fnmatch.filter(os.listdir(point_dir), 'mo*MOLog')[0]
        os.remove(join(point_dir, path))

        # Compute new guess at point k
        calc_new_wf_guess_on_points = [k]
        return compute_orbitals(
            None, package_name, project_name, path_hdf5,
            package_args, guess_args, package_config,
            calc_new_wf_guess_on_points, point_dir, job_files, k, gs)
    
    return promise_qm

def create_point_folder(work_dir, n, enumerate_from):
    """
    Create a new folder for each point in the MD trajectory.

    :returns: Paths lists.
    """
    folders = []
    for k in range(enumerate_from, n + enumerate_from):
        new_dir = join(work_dir, 'point_{}'.format(k))
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        folders.append(new_dir)

    return folders


def split_file_geometries(pathXYZ):
    """
    Reads a set of molecular geometries in xyz format and returns
    a list of string, where is element a molecular geometry

    :returns: String list containing the molecular geometries.
    """
    # Read Cartesian Coordinates
    with open(pathXYZ) as f:
        xss = f.readlines()

    numat = int(xss[0].split()[0])
    return list(map(''.join, chunksOf(xss, numat + 2)))


def create_dict_CGFs(path_hdf5, basisname, xyz, package_name='cp2k',
                     package_config=None):
    """
    Try to read the basis from the HDF5 otherwise read it from a file and store
    it in the HDF5 file. Finally, it reads the basis Set from HDF5 and
    calculate the CGF for each atom.

    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    type path_hdf5: String
    :param basisname: Name of the Gaussian basis set.
    :type basisname: String
    :param xyz: List of Atoms.
    :type xyz: [nac.common.AtomXYZ]
    """
    functions = {'cp2k': cp2k2hdf5, 'turbomole': turbomole2hdf5}

    basis_location = join(package_name, 'basis')
    with h5py.File(path_hdf5) as f5:
        if basis_location not in f5:
            # Search Path to the file containing the basis set
            pathBasis = package_config["basis"]
            keyBasis = InputKey("basis", [pathBasis])
            # Store the basis sets
            functions[package_name](f5, [keyBasis])

        return createNormalizedCGFs(f5, basisname, package_name, xyz)


def create_file_names(work_dir, i):
    """
    Creates a namedTuple with the name of the 4 files used
    for each point in the trajectory

    :returns: Namedtuple containing the IO files
    """
    file_xyz = join(work_dir, 'coordinates_{}.xyz'.format(i))
    file_inp = join(work_dir, 'point_{}.inp'.format(i))
    file_out = join(work_dir, 'point_{}.out'.format(i))
    file_MO = join(work_dir, 'mo_coeff_{}.out'.format(i))

    return JobFiles(file_xyz, file_inp, file_out, file_MO)
