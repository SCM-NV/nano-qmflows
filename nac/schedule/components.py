__all__ = ["calculate_mos", "create_point_folder",
           "split_file_geometries"]

# ================> Python Standard  and third-party <==========
from collections import (defaultdict, namedtuple)
from noodles import (gather, schedule)
from os.path import join

import fnmatch
import h5py
import logging
import os
import shutil

# ==================> Internal modules <==========
from nac.schedule.scheduleCp2k import prepare_job_cp2k
from nac.common import (Matrix, is_data_in_hdf5, read_cell_parameters_as_array)
from qmflows.hdf5 import dump_to_hdf5
from qmflows.utils import chunksOf
from qmflows.warnings_qmflows import SCF_Convergence_Warning

# Tuple contanining file paths
JobFiles = namedtuple("JobFiles", ("get_xyz", "get_inp", "get_out", "get_MO"))

# Starting logger
logger = logging.getLogger(__name__)
# ==============================> Tasks <=====================================


def calculate_mos(config: dict) -> list:
    """
    Look for the MO in the HDF5 file if they do not exists calculate them by
    splitting the jobs in batches given by the ``restart_chunk`` variables.
    Only the first job is calculated from scratch while the rest of the
    batch uses as guess the wave function of the first calculation in
    the batch.

    The config dict contains:

    :param geometries: list of molecular geometries
    :param project_name: Name of the project used as root path for storing
    data in HDF5.
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param folders: path to the directories containing the MO outputs
    :param settings_main: Settings for the job to run.
    :param calc_new_wf_guess_on_points: Calculate a new Wave function guess in
    each of the geometries indicated. By Default only an initial guess is
    computed.
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD

    :returns: path to nodes in the HDF5 file to MO energies
              and MO coefficients.
    """
    # Read Cell parameters file
    general = config['cp2k_general_settings']
    file_cell_parameters = general["file_cell_parameters"]
    if file_cell_parameters is not None:
        array_cell_parameters = read_cell_parameters_as_array(file_cell_parameters)[1]

    # First calculation has no initial guess
    # calculate the rest of the jobs using the previous point as initial guess
    orbitals = []  # list to the nodes in the HDF5 containing the MOs
    guess_job = None
    for j, gs in enumerate(config.geometries):

        # number of the point with respect to all the trajectory
        k = j + config.enumerate_from

        # dictionary containing the information of the j-th job
        dict_input = defaultdict(lambda: None)
        dict_input["geometry"] = gs
        dict_input["k"] = k

        # Path where the MOs will be store in the HDF5
        root = join(config.project_name, 'point_{}'.format(k), config.package_name, 'mo')
        dict_input["node_paths"] = [join(root, 'eigenvalues'), join(root, 'coefficients')]

        # If the MOs are already store in the HDF5 format return the path
        # to them and skip the calculation
        if is_data_in_hdf5(config.path_hdf5, dict_input["node_paths"]):
            logger.info("point_{} has already been calculated".format(k))
            orbitals.append(dict_input["node_paths"])
        else:
            logger.info("point_{} has been scheduled".format(k))

            # Add cell parameters from file if given
            if file_cell_parameters is not None:
                adjust_cell_parameters(general, array_cell_parameters, j)
            # Path to I/O files
            dict_input["point_dir"] = config.folders[j]
            dict_input["job_files"] = create_file_names(dict_input["point_dir"], k)
            dict_input["job_name"] = 'point_{}'.format(k)

            # Compute the MOs and return a new guess
            promise_qm = compute_orbitals(config, dict_input, guess_job)

            # Check if the job finishes succesfully
            promise_qm = schedule_check(promise_qm, config, dict_input)

            # Store the computation
            path_MOs = store_in_hdf5(config, dict_input, promise_qm)

            guess_job = promise_qm
            orbitals.append(path_MOs)

    return gather(*orbitals)


@schedule
def store_in_hdf5(config: dict, dict_input: dict, promise_qm: object) -> str:
    """
    Store the MOs in the HDF5
    """
    # Molecular Orbitals
    mos = promise_qm.orbitals
    if mos is not None:
        # Store in the HDF5
        try:
            with h5py.File(config.path_hdf5, 'r+') as f5:
                dump_to_hdf5(
                    mos, 'cp2k', f5,
                    project_name=config.project_name, job_name=dict_input["job_name"])
        # Remove the ascii MO file
        finally:
            work_dir = promise_qm.archive['work_dir']
            path_MOs = fnmatch.filter(os.listdir(work_dir), 'mo_*MOLog')[0]
            os.remove(join(work_dir, path_MOs))

    return dict_input["node_paths"]


def compute_orbitals(config: dict, dict_input: dict, guess_job) -> list:
    """
    Call a Quantum chemisty package to compute the MOs required to calculate
    the nonadiabatic coupling. When finish store the MOs in the HdF5 and
    returns a new guess.
    """

    dict_input["job_files"] = create_file_names(dict_input["point_dir"], dict_input["k"])

    # Calculating initial guess
    compute_guess = config.calc_new_wf_guess_on_points is not None

    # A job  is a restart if guess_job is None and the list of
    # wf guesses are not empty
    is_restart = guess_job is None and compute_guess

    pred = (dict_input['k'] in config.calc_new_wf_guess_on_points) or is_restart

    general = config.cp2k_general_settings

    if pred:
        guess_job = prepare_job_cp2k(
            general["cp2k_settings_guess"], dict_input, guess_job)

    promise_qm = prepare_job_cp2k(general["cp2k_settings_main"], dict_input, guess_job)

    return promise_qm


@schedule
def schedule_check(
        promise_qm: object, config: dict, dict_input: dict) -> object:
    """
    Check wether a calculation finishes succesfully otherwise run a new guess.
    """
    job_name = dict_input["job_name"]
    point_dir = dict_input["point_dir"]

    # Warnings of the computation
    warnings = promise_qm.warnings

    # Check for SCF convergence errors
    if not config.ignore_warnings and warnings is not None and any(
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
        config.calc_new_wf_guess_on_points.append(dict_input["k"])
        return compute_orbitals(config, dict_input, None)
    else:
        return promise_qm


def create_point_folder(work_dir, n, enumerate_from):
    """
    Create a new folder for each point in the MD trajectory.

    :returns: Paths lists.
    """
    folders = []
    for k in range(enumerate_from, n + enumerate_from):
        new_dir = join(work_dir, 'point_{}'.format(k))
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir)
        folders.append(new_dir)

    return folders


def split_file_geometries(pathXYZ: str) -> list:
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


def create_file_names(work_dir: str, i: int):
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


def adjust_cell_parameters(general: dict, array_cell_parameters: Matrix, j: int) -> None:
    """
    If the cell parameters change during the MD simulations, adjust them
    for the molecular orbitals computation
    """
    for s in (general[p] for p in ('cp2k_settings_main', 'cp2k_settings_guess')):
        s.cell_parameters = array_cell_parameters[j, 2:11].reshape(3, 3).tolist()
        s.cell_angles = None
