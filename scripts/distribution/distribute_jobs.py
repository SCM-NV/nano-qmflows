
from collections import namedtuple
from nac.workflows.initialization import split_trajectory
from os.path import join
from qmworks import Settings
from qmworks.utils import settings2Dict

import getpass
import os
import shutil
import string
import subprocess

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
    project_name = 'distribute_Cd33Se33'  # name use to create folders

    # Path to the basis set used by Cp2k
    home = os.path.expanduser('~')
    basisCP2K = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potCP2K = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")
    lower_orbital, upper_orbital = 278, 317
    cp2k_main, cp2k_guess = cp2k_input(lower_orbital, upper_orbital,
                                       cell_parameters=28)

    # Trajectory splitting
    path_to_trajectory = "traj1000.xyz"
    blocks = 5  # Number of chunks to split the trajectory

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
    main_dft.scf.eps_scf = 1e-5
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
    ot_dft.scf.eps_scf = 1e-05
    ot_dft.scf.scf_guess = 'restart'
    cp2k_OT.specific.cp2k.force_eval.subsys.cell.periodic = 'None'

    return cp2k_args, cp2k_OT


# ============================> Distribution <=================================


def distribute_computations(project_name, basisCP2K, potCP2K, cp2k_main,
                            cp2k_guess, path_to_trajectory, blocks, slurm):

    script_name = "script_remote_function.py"
    # Split the trajectory in Chunks and move each chunk to its corresponding
    # directory.
    chunks_trajectory = split_trajectory(path_to_trajectory, blocks, '.')
    chunks_trajectory.sort()
    enumerate_from = 0
    for  file_xyz, l in zip(chunks_trajectory, string.ascii_lowercase):
        folder = 'chunk_{}'.format(l)
        os.mkdir(folder)
        shutil.move(file_xyz, folder)
        # function to be execute remotely
        write_python_script(folder, file_xyz, project_name,
                            basisCP2K, potCP2K, cp2k_main,
                            cp2k_guess, enumerate_from, script_name)
        write_slurm_script(folder, slurm, script_name)
        enumerate_from += number_of_geometries(join(folder, file_xyz))


def write_python_script(folder, file_xyz, project_name, basisCP2K, potCP2K, cp2k_main,
                        cp2k_guess, enumerate_from, script_name):
    """ Write the python script to compute the PYXAID hamiltonians"""
    scratch = '/scratch-shared'
    user = getpass.getuser()
    path_hdf5 = join(scratch, user, project_name, '{}.hdf5'.format(folder))

    xs = """
from nac.workflows.workflow_coupling import generate_pyxaid_hamiltonians
from nac.workflows.initialization import initialize
from qmworks.utils import dict2Setting
import plams

plams.init()

project_name = '{}'
path_basis = '{}'
path_potential = '{}'
path_hdf5 = '{}'
path_traj_xyz = '{}'
basisname = '{}'

initial_config = initialize(project_name, path_traj_xyz,
                            basisname=basisname,
                            path_basis=path_basis,
                            path_potential=path_potential,
                            enumerate_from={},
                            calculate_guesses='first',
                            path_hdf5=path_hdf5)

cp2k_main = dict2Setting({})
cp2k_guess = dict2Setting({})

generate_pyxaid_hamiltonians('cp2k', project_name, cp2k_main,
                             guess_args=cp2k_guess, nCouplings=40,
                             **initial_config)
plams.finish()
 """.format(project_name, basisCP2K, potCP2K, path_hdf5, file_xyz, cp2k_main.basis,
            enumerate_from, settings2Dict(cp2k_main), settings2Dict(cp2k_guess))

    with open(join(folder, script_name), 'w') as f:
        f.write(xs)

    return script_name


def write_slurm_script(folder, slurm, python_script):
    """
    write an Slurm launch script
    """
    sbatch = lambda x, y: "#SBATCH -{} {}\n".format(x, y)

    header = "#! /bin/bash\n"
    modules = "\nmodule load cp2k/3.0\nsource activate qmworks\n\n"
    time = sbatch('t', slurm.time)
    nodes = sbatch('N', slurm.nodes)
    tasks = sbatch('n', slurm.tasks)
    name = sbatch('J', slurm.name)
    python = "python {}".format(python_script)

    # Script content
    content = header + time + nodes + tasks + name + modules + python

    file_name = join(folder, "launch.sh")

    with open(file_name, 'w') as f:
        f.write(content)
    return file_name


def number_of_geometries(file_name):
    """
    Count the number of geometries in XYZ formant in a given file.
    """

    with open(file_name, 'r') as f:
        numat = int(f.readline())

    cmd = "wc -l {}".format(file_name)
    wc = subprocess.getoutput(cmd).split()[0]

    lines_per_geometry = numat + 2

    return int(wc) // lines_per_geometry


if __name__ == "__main__":
    main()
