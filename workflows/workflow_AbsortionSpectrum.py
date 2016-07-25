
# ================> Python Standard  and third-party <==========
from noodles import gather, schedule
from qmworks import run, Settings
from qmworks.parsers import parse_string_xyz
from os.path import join

import getpass
import numpy as np
import os
import plams
import shutil

# =========================> Internal modules <================================
from nac.common import retrieve_hdf5_data
from nac.integrals.absorptionSpectrum import oscillator_strength
from nac.schedule.components import (calculate_mos, create_dict_CGFs,
                                     create_point_folder, split_file_geometries)
from nac.schedule.scheduleCoupling import  schedule_transf_matrix

# ==============================> Main <==================================


def simulate_absoprtion_spectrum(package_name, project_name, geometry,
                                 package_args, guess_args=None,
                                 initial_states=None, final_states=None,
                                 calc_new_wf_guess_on_points=[0],
                                 path_hdf5=None, package_config=None):
    """
    Compute the oscillator strength

    :param package_name: Name of the package to run the QM simulations.
    :type  package_name: String
    :param project_name: Folder name where the computations
    are going to be stored.
    :type project_name: String
    :param all_geometries:string containing the molecular geometries
    numerical results.
    :type path_traj_xyz: String
    :param package_args: Specific settings for the package
    :type package_args: dict
    :param package_args: Specific settings for guess calculate with `package`.
    :type package_args: dict
    :param initial_states: List of the initial Electronic states.
    :type initial_states: [Int]
    :param final_states: List containing the sets of possible electronic states.
    :type final_states: [[Int]]
    :param calc_new_wf_guess_on_points: Points where the guess wave functions
    are calculated.
    :type use_wf_guess_each: [Int]
    :param package_config: Parameters required by the Package.
    :type package_config: Dict
    :returns: None
    """
    #  Environmental Variables
    cwd = os.path.realpath(".")
    
    basisName = package_args.basis
    work_dir = os.path.join(cwd, project_name)
    if path_hdf5 is None:
        path_hdf5 = os.path.join(work_dir, "quantum.hdf5")

    # Create Work_dir if it does not exist
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)

    # Generate a list of tuples containing the atomic label
    # and the coordinates to generate
    # the primitive CGFs
    atoms = parse_string_xyz(geometry[0])
    dictCGFs = create_dict_CGFs(path_hdf5, basisName, atoms, package_config)

    # Calculcate the matrix to transform from cartesian to spherical
    # representation of the overlap matrix
    hdf5_trans_mtx = schedule_transf_matrix(path_hdf5, atoms,
                                            basisName, project_name,
                                            packageName=package_name)

    # Create a folder for each point the the dynamics
    traj_folders = create_point_folder(work_dir, 1, 0)

    # prepare Cp2k Jobs
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(package_name, geometry, project_name,
                                  path_hdf5, traj_folders, package_args,
                                  guess_args, calc_new_wf_guess_on_points=[0],
                                  enumerate_from=0,
                                  package_config=package_config)
    
    oscillators = calcOscillatorStrenghts(project_name, mo_paths_hdf5, dictCGFs,
                                          geometry, path_hdf5, mo_paths_hdf5[0],
                                          hdf5_trans_mtx=hdf5_trans_mtx,
                                          initial_states=initial_states,
                                          final_states=final_states)
    rs = run(oscillators)
    print(rs)


def calcOscillatorStrenghts(project_name, mo_paths_hdf5, dictCGFs, geometry,
                            path_hdf5, mo_path_hdf5, hdf5_trans_mtx=None,
                            initial_states=None, final_states=None):
    """
    Use the Molecular orbital Energies and Coefficients to compute the
    oscillator_strength.
    """
    cgfsN = [dictCGFs[x.symbol] for x in geometry]

    es, coeffs = retrieve_hdf5_data(path_hdf5, mo_path_hdf5[0])

    # If the MO orbitals are given in Spherical Coordinates transform then to
    # Cartesian Coordinates.
    trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx) if hdf5_trans_mtx else None

    oscillators = []
    for initialS, fs in zip(initial_states, final_states):
        css_i = coeffs[:, initialS]
        energy_i = es[initialS]
        for finalS in fs:
            css_j = coeffs[:, initialS]
            energy_j = es[finalS]
            deltaE = energy_j - energy_i
            fij = callScheduleOsc(geometry, cgfsN, css_i, css_j, deltaE,
                                  trans_mtx=trans_mtx)
            oscillators.append(fij)

    return oscillators


def callScheduleOsc(geometry, cgfsN, css, energy, hdf5_trans_mtx=None):
    """
    """
    scheduleOscillatorStrength = schedule(oscillator_strength)
    sh, = coeffs.shape
    css = np.tile(coeffs, sh)
    
    if hdf5_trans_mtx is not None:
        transpose = np.transpose(trans_mtx)
        css = np.dot(trans_mtx, np.dot(css, transpose))  # Overlap in Sphericals
    

  
        
        
        fij = scheduleOscillatorStrength(geometry, cgfsN, css, energy)
        
# ===================================<>========================================


def main():
    """
    Initialize the arguments to compute the nonadiabatic coupling matrix for
    a given MD trajectory.
    """
    initial_states = []
    final_states = [[]]
    
    plams.init()
    project_name = 'spectrum_pentacene'

    cell = [[16.11886919, 0.07814137, -0.697284243],
            [-0.215317662, 4.389405268, 1.408951791],
            [-0.216126961, 1.732808365, 9.748961085]]
    # create Settings for the Cp2K Jobs
    cp2k_args = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = cell
    cp2k_args.specific.cp2k.force_eval.dft.scf.added_mos = 100
    cp2k_args.specific.cp2k.force_eval.dft.scf.diagonalization.jacobi_threshold = 1e-6

    # Setting to calculate the WF use as guess
    cp2k_OT = Settings()
    cp2k_OT.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_OT.potential = "GTH-PBE"
    cp2k_OT.cell_parameters = cell
    cp2k_OT.specific.cp2k.force_eval.dft.scf.scf_guess = 'atomic'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.minimizer = 'DIIS'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.n_diis = 7
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.added_mos = 0
    cp2k_OT.specific.cp2k.force_eval.dft.scf.eps_scf = 5e-06

    # Path to the MD geometries
    path_traj_xyz = "trajectory.xyz"

    # User variables
    home = os.path.expanduser('~')  # HOME Path
    username = getpass.getuser()
    
    # Work_dir
    scratch = "/scratch-shared"
    scratch_path = join(scratch, username, project_name)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # Cp2k configuration files
    basiscp2k = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potcp2k = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")
    cp2k_config = {"basis": basiscp2k, "potential": potcp2k}

    # HDF5 path
    path_hdf5 = join(scratch_path, 'quantum.hdf5')

    # all_geometries type :: [String]
    geometry = split_file_geometries(path_traj_xyz)

    # Calculate new Guess in each Geometry
    pointsGuess = [0]

    # Hamiltonian computation
    simulate_absoprtion_spectrum('cp2k', project_name, geometry, cp2k_args,
                                 guess_args=cp2k_OT,
                                 initial_states=initial_states,
                                 final_states=final_states,
                                 calc_new_wf_guess_on_points=pointsGuess,
                                 path_hdf5=path_hdf5,
                                 package_config=cp2k_config)


# ===================================<>========================================
if __name__ == "__main__":
    main()
