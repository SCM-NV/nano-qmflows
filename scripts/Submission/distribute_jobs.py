
from collections import namedtuple
from nac.workflows.workflow_coupling import generate_pyxaid_hamiltonians
from nac.workflows.initialization import (initialize, split_trajectory)
from os.path import join
from qmworks import Settings

import os
import plams


SLURM = namedtuple("SLURM", ("nodes", "tasks", "time", "name"))


def main():
    """
    THE USER MUST CHANGES THESE VARIABLES ACCORDING TO HER/HIS NEEDS:
      * project_name
      * path to the basis and Cp2k Potential
      * CP2K:
          - Range of Molecular oribtals printed by CP2K
          - Cell parameter
      * Settings to Run Cp2k simulations
      * Path to the trajectory in XYZ

    The slurm configuration is optional but the user can edit it:
        property  default
       * nodes         2
       * tasks        24
       * time   48:00:00
       * name       namd

    """
    # USER DEFINED CONFIGURATION
    project_name = 'test_Cd33Se33'  # name use to create folders

    # Path to the basis set used by Cp2k
    home = os.path.expanduser('~')
    basisCP2K = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potCP2K = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")
    lower_orbital, upper_orbital = None, None
    cp2k_main, cp2k_guess = cp2k_input(lower_orbital, upper_orbital,
                                       cell_parameters=50)

    # Trajectory splitting
    path_to_trajectory = "PATH/to/xyz"
    blocks = None  # Number of chunks to split the trajectory
    
    # SLURM Configuration
    slurm = SLURM(
        nodes=2,
        tasks=24,
        time="48:00:00",
        name="namd"
    )

    distribute_computations(project_name, basisCP2K, potCP2K, cp2k_main,
                            cp2k_guess, path_to_trajectory, blocks, slurm)


def cp2k_input(lower_orbital, upper_orbital, cell_parameters=None):
    """
    # create ``Settings`` for the Cp2K Jobs.
    """
    # Main Cp2k Jobs
    cp2k_args = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = [cell_parameters] * 3
    main_dft = cp2k_args.specific.cp2k.force_eval.dft
    main_dft.scf.added_mos = 20
    main_dft.scf.max_scf = 200
    main_dft['print']['mo']['mo_index_range'] = "{} {}".format(lower_orbital,
                                                               upper_orbital)
    cp2k_args.specific.cp2k.force_eval.subsys.cell.periodic = 'None'

    # Setting to calculate the wave function used as guess
    cp2k_OT = Settings()
    cp2k_OT.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_OT.potential = "GTH-PBE"
    cp2k_OT.cell_parameters = [cell_parameters] * 3
    ot_dft = cp2k_OT.specific.cp2k.force_eval.dft
    ot_dft.scf.scf_guess = 'atomic'
    ot_dft.scf.ot.minimizer = 'DIIS'
    ot_dft.scf.ot.n_diis = 7
    ot_dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
    ot_dft.scf.added_mos = 0
    ot_dft.scf.eps_scf = 5e-06
    ot_dft.scf.scf_guess = 'restart'
    cp2k_OT.specific.cp2k.force_eval.subsys.cell.periodic = 'None'

    return cp2k_args, cp2k_OT


# ============================> Distribution <=================================


def distribute_computations(path_to_trajectory, blocks):

    chunks_trajectory = split_trajectory(path_to_trajectory, blocks, '.')


def remote_fuction(project_name, path_traj_xyz, basisCP2K, potCP2K, cp2k_main,
                   cp2k_guess):
    plams.init()

    basisname = cp2k_main.basis

    initial_config = initialize(project_name, path_traj_xyz,
                                basisname=basisname, path_basis=basisCP2K,
                                path_potential=potCP2K,
                                enumerate_from=0,
                                calculate_guesses='first',
                                scratch='/scratch-shared')

    generate_pyxaid_hamiltonians('cp2k', project_name, cp2k_main,
                                 guess_args=cp2k_guess, nCouplings=40,
                                 **initial_config)
    plams.finish()


if __name__ == "__main__":
    main()
