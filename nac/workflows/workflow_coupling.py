__author__ = "Felipe Zapata"

__all__ = ['generate_pyxaid_hamiltonians']

# ================> Python Standard  and third-party <==========
from os.path import join

import os
import plams

# =========================> Internal modules <================================
from noodles import gather, schedule

from nac import initialize
from nac.common import change_mol_units
from nac.schedule.components import calculate_mos
from nac.schedule.scheduleCoupling import (lazy_schedule_couplings,
                                           write_hamiltonians)
from qmworks import run, Settings
from qmworks.parsers import parse_string_xyz

# ==============================> Main <==================================


def generate_pyxaid_hamiltonians(package_name, project_name,
                                 cp2k_args, guess_args=None,
                                 path=None,
                                 geometries=None, dictCGFs=None,
                                 calc_new_wf_guess_on_points=None,
                                 path_hdf5=None, enumerate_from=0,
                                 package_config=None, dt=1,
                                 traj_folders=None, work_dir=None,
                                 basisname=None, hdf5_trans_mtx=None,
                                 nHOMO=None, couplings_range=None):
    """
    Use a md trajectory to generate the hamiltonian components to tun PYXAID
    nmad.

    :param package_name: Name of the package to run the QM simulations.
    :type  package_name: String
    :param project_name: Folder name where the computations
    are going to be stored.
    :type project_name: String
    :param geometries: List of string cotaining the molecular geometries
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
    :param nHOMO: index of the HOMO orbital.
    :param couplings_range: Range of MO use to compute the nonadiabatic
    coupling matrix.
    :returns: None
    """
    # prepare Cp2k Jobs
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(package_name, geometries, project_name,
                                  path_hdf5, traj_folders, cp2k_args,
                                  guess_args, calc_new_wf_guess_on_points,
                                  enumerate_from,
                                  package_config=package_config)

    # Calculate Non-Adiabatic Coupling
    # Number of Coupling points calculated with the MD trajectory
    nPoints = len(geometries) - 2
    promise_couplings = [calculate_coupling(i, path_hdf5, dictCGFs,
                                            geometries,
                                            mo_paths_hdf5, hdf5_trans_mtx,
                                            enumerate_from,
                                            output_folder=project_name,
                                            nHOMO=nHOMO,
                                            couplings_range=couplings_range,
                                            dt=dt, units='angstrom')
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

    hams_files = run(promise_files, path=path)

    print(hams_files)
# ==============================> Tasks <=====================================


def calculate_coupling(i, path_hdf5, dictCGFs, all_geometries, mo_paths,
                       hdf5_trans_mtx, enumerate_from, output_folder=None,
                       nHOMO=None, couplings_range=None,
                       dt=1, units='angstrom'):
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
    :type all_geometries: [String]
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
    xss = all_geometries[i], all_geometries[j], all_geometries[k]

    geometries = tuple(map(parse_string_xyz, xss))

    if 'angstrom' in units.lower():
        geometries = tuple(map(change_mol_units, geometries))

    return lazy_schedule_couplings(i, path_hdf5, dictCGFs, geometries, mo_paths,
                                   dt=dt, hdf5_trans_mtx=hdf5_trans_mtx,
                                   output_folder=output_folder,
                                   enumerate_from=enumerate_from,
                                   nHOMO=nHOMO, couplings_range=couplings_range)
# ============<>===============


def main():
    """
    Initialize the arguments to compute the nonadiabatic coupling matrix for
    a given MD trajectory.
    """
    plams.init()

    # create Settings for the Cp2K Jobs
    cp2k_args = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = [50.0] * 3
    main_dft = cp2k_args.specific.cp2k.force_eval.dft
    main_dft.scf.added_mos = 100
    main_dft.scf.diagonalization.jacobi_threshold = 1e-6

    # Setting to calculate the WF use as guess
    cp2k_OT = Settings()
    cp2k_OT.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_OT.potential = "GTH-PBE"
    cp2k_OT.cell_parameters = [50.0] * 3
    ot_dft = cp2k_OT.specific.cp2k.force_eval.dft
    ot_dft.scf.scf_guess = 'atomic'
    ot_dft.scf.ot.minimizer = 'DIIS'
    ot_dft.scf.ot.n_diis = 7
    ot_dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
    ot_dft.scf.added_mos = 0
    ot_dft.scf.eps_scf = 5e-06

    # project
    project_name = 'ET_Pb79S44'
    path_traj_xyz = "./trajectory_4000-5000.xyz"
    basisname = cp2k_args.basis

    home = os.path.expanduser('~')
    basiscp2k = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potcp2k = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")

    initial_config = initialize(project_name, path_traj_xyz,
                                basisname=basisname, path_basis=basiscp2k,
                                path_potential=potcp2k,
                                enumerate_from=0,
                                calculate_guesses='first')

    # Hamiltonian computation
    generate_pyxaid_hamiltonians('cp2k', project_name, cp2k_args,
                                 guess_args=cp2k_OT, nCouplings=40,
                                 **initial_config)
    plams.finish()

if __name__ == "__main__":
    main()
