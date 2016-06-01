__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from collections import namedtuple
from functools import reduce, partial
from os.path import join

import h5py
import os
import plams

# ==================> Internal modules <==========
from nac.basisSet.basisNormalization import createNormalizedCGFs
from noodles import gather, schedule

from qmworks.fileFunctions import search_environ_var
from qmworks import run, Settings
from qmworks.common import InputKey
from qmworks.hdf5.quantumHDF5 import cp2k2hdf5
from qmworks.parsers import parse_string_xyz
from qmworks.utils import chunksOf, flatten
from nac.schedule.scheduleCoupling import (lazy_schedule_couplings,
                                           schedule_transf_matrix,
                                           write_hamiltonians)
from nac.schedule.scheduleCp2k import prepare_job_cp2k

# ==============================<>=========================
# Tuple contanining file paths
JobFiles = namedtuple("JobFiles", ("get_xyz", "get_inp", "get_out", "get_MO"))

# ==============================> Main <==================================


def generate_pyxaid_hamiltonians(package_name, project_name, all_geometries,
                                 cp2k_args, guess_args=None,
                                 calc_new_wf_guess_on_points=[0],
                                 path_hdf5=None, enumerate_from=0):
    """
    Use a md trajectory to generate the hamiltonian components to tun PYXAID
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
    :returns: None
    """
    #  Environmental Variables
    cwd = os.path.realpath(".")
    
    basisName = cp2k_args.basis
    work_dir = os.path.join(cwd, project_name)
    if path_hdf5 is None:
        path_hdf5 = os.path.join(work_dir, "quantum.hdf5")

    # Create Work_dir if it does not exist
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # Generate a list of tuples containing the atomic label
    # and the coordinates to generate
    # the primitive CGFs
    atoms = parse_string_xyz(all_geometries[0])
    dictCGFs = create_dict_CGFs(path_hdf5, basisName, atoms)

    # Calculcate the matrix to transform from cartesian to spherical
    # representation of the overlap matrix
    hdf5_trans_mtx = schedule_transf_matrix(path_hdf5, atoms,
                                            basisName, work_dir,
                                            packageName=package_name)

    # Create a folder for each point the the dynamics
    traj_folders = create_point_folder(work_dir, len(all_geometries),
                                       enumerate_from)

    # prepare Cp2k Jobs
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(package_name, all_geometries, work_dir,
                                  path_hdf5, traj_folders, cp2k_args,
                                  guess_args, calc_new_wf_guess_on_points,
                                  enumerate_from)

    # Calculate Non-Adiabatic Coupling
    # Number of Coupling points calculated with the MD trajectory
    nPoints = len(all_geometries) - 2
    promise_couplings = [calculate_coupling(i, path_hdf5, dictCGFs,
                                            all_geometries,
                                            mo_paths_hdf5, hdf5_trans_mtx,
                                            enumerate_from,
                                            output_folder=work_dir)
                         for i in range(nPoints)]
    path_couplings = gather(*promise_couplings)

    # Write the results in PYXAID format
    path_hamiltonians = join(work_dir, 'hamiltonians')
    if not os.path.exists(path_hamiltonians):
        os.makedirs(path_hamiltonians)

    # Inplace scheduling of write_hamiltonians function.
    # Equivalent to add @schedule on top of the function
    schedule_write_ham = schedule(write_hamiltonians)
    
    promise_files = schedule_write_ham(path_hdf5, work_dir, mo_paths_hdf5,
                                       path_couplings, nPoints,
                                       path_dir_results=path_hamiltonians,
                                       enumerate_from=enumerate_from)

    hams_files = run(promise_files)

    print(hams_files)
# ==============================> Tasks <=====================================


def calculate_mos(package_name, all_geometries, work_dir, path_hdf5, folders,
                  package_args, guess_args=None, calc_new_wf_guess_on_points=[0],
                  enumerate_from=0):
    """
    Look for the MO in the HDF5 file if they do not exists calculate them by
    splitting the jobs in batches given by the ``restart_chunk`` variables.
    Only the first job is calculated from scratch while the rest of the
    batch uses as guess the wave function of the first calculation in
    the batch.

    :param all_geometries: list of molecular geometries
    :type all_geometries: String list
    :param work_dir: Path to the work directory
    :type work_dir: String
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :type path_hdf5: String
    :param folders: path to the directories containing the MO outputs
    :type folders: String list
    :param calc_new_wf_guess_on_points: Calculate a new Wave function guess in
    each of the geometries indicated. By Default only an initial guess is
    computed.
    :type calc_new_wf_guess_on_points: [Int]
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :type enumerate_from: Int
    :returns: path to nodes in the HDF5 file to MO energies and MO coefficients.
    """
    def create_properties_path(i):
        """
        Path inside HDF5 where the data is stored
        """
        rs = join(work_dir, 'point_{}'.format(i), package_name, 'mo')
        return [join(rs, 'eigenvalues'), join(rs, 'coefficients')]

    def search_data_in_hdf5(i):
        """
        Search if the node exists in the HDF5 file.
        """
        paths_to_prop = create_properties_path(i)

        with h5py.File(path_hdf5, 'r') as f5:
            if isinstance(paths_to_prop, list):
                pred = all(path in f5 for path in paths_to_prop)
            else:
                pred = paths_to_prop in f5

        return paths_to_prop if pred else  None

    path_to_orbitals = []

    # Calculating initial guess
    point_dir = folders[0]
    job_files = create_file_names(point_dir, 0)
    # calculate the rest of the job using the previous point as initial guess
    guess_job = None
    for j, gs in enumerate(all_geometries):
        k = j + enumerate_from
        if k in calc_new_wf_guess_on_points:
            guess_job = call_schedule_qm(package_name, guess_args, path_hdf5,
                                         point_dir, job_files, k,
                                         all_geometries[k], guess_job=guess_job,
                                         store_in_hdf5=False)
        point_dir = folders[j]
        job_files = create_file_names(point_dir, k)
        paths_to_prop = search_data_in_hdf5(k)

        # If the MOs are already store in the HDF5 format return the path
        # to them and skip the calculation
        if paths_to_prop is not None:
            path_to_orbitals.append(paths_to_prop)
        else:
            promise_qm = call_schedule_qm(package_name, package_args,
                                          path_hdf5, point_dir, job_files,
                                          k, gs, guess_job)
            path_to_orbitals.append(promise_qm.orbitals)
            guess_job = promise_qm
            
    return gather(*path_to_orbitals)


def call_schedule_qm(packageName, package_args, path_hdf5, point_dir,
                     job_files, k, geometry, guess_job=None,
                     store_in_hdf5=True):
    """
    Call an external computational chemistry software to do some calculations

    :param package_name: Name of the package to run the QM simulations.
    :type  package_name: String
    :param package_args: Specific settings for the package
    :type package_args: Settings
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    type path_hdf5: String
    :param point_dir: path to the directory where the output is written.
    :type point_dir: String
    :param job_files: Tuple containing the absolute path to IO files.
    :type job_files: NamedTuple
    :param k: current point being calculate in the MD
    :type k: Int
    :param geometry: Molecular geometry
    :type geometry: String
    """
    prepare_and_schedule = {'cp2k': prepare_job_cp2k}
    
    job = prepare_and_schedule[packageName](geometry, job_files, package_args, k,
                                            point_dir, hdf5_file=path_hdf5,
                                            wfn_restart_job=guess_job,
                                            store_in_hdf5=store_in_hdf5)

    return job


def calculate_coupling(i, path_hdf5, dictCGFs, all_geometries, mo_paths,
                       hdf5_trans_mtx, enumerate_from, output_folder=None):
    """
    Calculate the non-adiabatic coupling using 3 consecutive set of MOs in
    a dynamics. Explicitly declares that each Coupling Depends in
    three set of MOs.

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
    :param mo_paths: Path to the MO coefficients and energies in the
    HDF5 file.
    :type mo_paths: [String]
    :param hdf5_trans_mtx: path to the transformation matrix in the HDF5 file.
    :type hdf5_trans_mtx: String
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :type enumerate_from: Int
    :returns: promise to path to the Coupling inside the HDF5
    """
    j, k = i + 1, i + 2
    geometries = all_geometries[i], all_geometries[j], all_geometries[k]

    return lazy_schedule_couplings(i, path_hdf5, dictCGFs, geometries, mo_paths,
                                   dt=1, hdf5_trans_mtx=hdf5_trans_mtx,
                                   output_folder=output_folder,
                                   enumerate_from=enumerate_from)


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
    return list(map(flatten, chunksOf(xss, numat + 2)))


#FIXME: Extended to other QM packages
def create_dict_CGFs(path_hdf5, basisname, xyz):
    """
    If the Cp2k Basis are already stored in the hdf5 file continue,
    otherwise read and store them in the hdf5 file.
    """
    # Try to read the basis otherwise read it from a file
    with h5py.File(path_hdf5, chunks=True) as f5:
        try:
            f5["cp2k/basis"]
        except KeyError:
            # Search Path to the file containing the basis set
            pathBasis = search_environ_var('BASISCP2K')
            keyBasis = InputKey("basis", [pathBasis])
            cp2k2hdf5(f5, [keyBasis])             # Store the basis sets
        # Read the basis Set from HDF5 and calculate the CGF for each atom
        dictCGFs = createNormalizedCGFs(f5, basisname, 'cp2k', xyz)

    return dictCGFs


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

# ============<>===============


# Example of Workflow
def main():
    plams.init()
    project_name = 'ET_Pb79S44'

    # create Settings for the Cp2K Jobs
    cp2k_args = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = [50.0] * 3
    cp2k_args.specific.cp2k.force_eval.dft.scf.added_mos = 100
    cp2k_args.specific.cp2k.force_eval.dft.scf.diagonalization.jacobi_threshold = 1e-6

    # Setting to calculate the WF use as guess
    cp2k_OT = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = [50.0] * 3
    cp2k_OT.specific.cp2k.force_eval.dft.scf.scf_guess = 'atomic'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.minimizer = 'DIIS'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.n_diis = 7
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.added_mos = 0
    cp2k_OT.specific.cp2k.force_eval.dft.scf.eps_scf = 5e-06

    # Path to the MD geometries
    path_traj_xyz = "./trajectory_4000-5000.xyz"

    # Work_dir
    scratch = "/scratch-shared"
    scratch_path = join(scratch, project_name)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # HDF5 path
    path_hdf5 = join(scratch_path, 'quantum.hdf5')

    # all_geometries type :: [String]
    geometries = split_file_geometries(path_traj_xyz)

    # Named the points of the MD starting from this number
    enumerate_from = 0

    # Calculate new Guess in each Geometry
    pointsGuess = [enumerate_from + i for i in range(len(geometries))]

    # Hamiltonian computation
    generate_pyxaid_hamiltonians('cp2k', project_name, geometries, cp2k_args,
                                 setting_guess=cp2k_OT,
                                 calc_new_wf_guess_on_points=pointsGuess,
                                 path_hdf5=path_hdf5,
                                 enumerate_from=enumerate_from)

    print("PATH TO HDF5:{}\n".format(path_hdf5))
    plams.finish()

if __name__ == "__main__":
    main()

