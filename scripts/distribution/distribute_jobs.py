
from collections import namedtuple
from nac.workflows.initialization import split_trajectory
from os.path import join
from qmworks import Settings
from qmworks.utils import settings2Dict

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

    # Algorithm use to compute the derivative coupling
    # Either levine or 3points
    algorithm='levine'


    # Varaible to define the Path ehere the Cp2K jobs will be computed
    scratch = "<Path/where/the/Molecular_Orbitals/and/Couplings/are/computed>"
    project_name = 'replace_with_Name_of_the_Project'  # name use to create folders

    # Path to the basis set used by Cp2k
    basisCP2K = "<Path/to/the/BASIS_MOLOPT>"
    potCP2K = "<Path/to/the/GTH_POTENTIALS>"

    # Cell parameter can be a:
    # * number.
    # * list contaning 3 Number.
    # * list of list containing the matrix describing the cell.
    cell_parameters = None

    # Angles of the cell
    cell_angles = [90.0, 90.0, 90.0]

    # Range of Molecular orbitals use to print.
    # They will be used to compute the derivative coupling. If you want
    # to compute the coupling with 40 Orbitals: 20 occupied and 20 virtual,
    # you should choose the following index in such a way that you
    # print 20 HOMOs and 20 LUMOs.

    range_orbitals = Lower_Index, Highest_Index

    # In relation to the above range
    # The store orbitals store in the HDF5 are going to be reindex in
    # such way tha the Molecular corresponding with Lower_Index
    # is assigned the index 1.
    # Then, What is the index from the HOMO?
    # If you are asking for 100 orbitals, 50 HOMOs and 50 LUMOs then nHOMO = 50
    # If you are asking for 100 orbitals, 30 HOMOS and 70 LUMos then nHOMO = 30
    nHOMO = None

    # Given the range_orbitals define above, which of those orbitals are going
    # to be used to compute the nonadiabatic copling?

    # If you are asking for 100 orbitals of which 50 are HOMOs and 50 are
    # LUMOs, and you want to compute the coupling btween all of then
    # then coupling_range = (50, 50)

    # If you want to compute only a subset of the orbitals specify which orbitals
    # you want to use. For instance if nHOMO = 50, the value
    # coupling_range = (30, 20) means that you want to compute the coupling
    # between the 30 Highest Occupied Molecular Orbitals  and the 20
    # Lowest Unoccupied Molecular Orbitals
    coupling_range = None

    # The keyword added_mos takes as input the number of LUMOs that are needed
    # to compute the desired number of couplings
    added_mos = coupling_range[1]

    # Trajectory splitting
    path_to_trajectory = "<Path/to/the/trajectory/in/xyz/format"

    # Number of chunks to split the trajectory
    blocks = 5

    # Time step in femtoseconds use to compute the derivative coupling.
    # It corresponds with the integration step of the MD.
    dt = 1  # 1 femtosecond

    # SLURM Configuration
    slurm = SLURM(
        nodes=2,
        tasks=24,
        time="48:00:00",
        name="namd"
    )

    # Generate the CP2K inputs that will be then broadcasted to different nodes
    cp2k_main, cp2k_guess = cp2k_input(range_orbitals, cell_parameters,
                                       cell_angles, added_mos)

    # Path where the data will be copy back
    cwd = os.getcwd()

    distribute_computations(scratch, project_name, basisCP2K, potCP2K,
                            cp2k_main, cp2k_guess, path_to_trajectory, blocks,
                            slurm, cwd, nHOMO, coupling_range, algorithm, dt)


def cp2k_input(
        range_orbitals, cell_parameters, cell_angles, added_mos,
        basis="DZVP-MOLOPT-SR-GTH", potential="GTH-PBE"):
    """
    # create ``Settings`` for the Cp2K Jobs.
    """
    # Main Cp2k Jobs
    cp2k_args = Settings()
    cp2k_args.basis = fun_format(basis)
    cp2k_args.potential = fun_format(potential)
    cp2k_args.cell_parameters = cell_parameters
    cp2k_args.cell_angles = cell_angles
    main_dft = cp2k_args.specific.cp2k.force_eval.dft
    main_dft.scf.added_mos = added_mos
    main_dft.scf.max_scf = 40
    main_dft.scf.eps_scf = 5e-4
    main_dft['print']['mo']['mo_index_range'] = '"{} {}"'.format(*range_orbitals)
    cp2k_args.specific.cp2k.force_eval.subsys.cell.periodic = fun_format('None')

    # Setting to calculate the wave function used as guess
    cp2k_OT = Settings()
    cp2k_OT.basis = fun_format(basis)
    cp2k_OT.potential = fun_format(potential)
    cp2k_OT.cell_parameters = cell_parameters
    cp2k_OT.cell_angles = cell_angles
    ot_dft = cp2k_OT.specific.cp2k.force_eval.dft
    ot_dft.scf.scf_guess = fun_format('atomic')
    ot_dft.scf.ot.minimizer = fun_format('DIIS')
    ot_dft.scf.ot.n_diis = 7
    ot_dft.scf.ot.preconditioner = fun_format('FULL_SINGLE_INVERSE')
    ot_dft.scf.added_mos = 0
    ot_dft.scf.eps_scf = 1e-06
    ot_dft.scf.scf_guess = fun_format('restart')
    cp2k_OT.specific.cp2k.force_eval.subsys.cell.periodic = fun_format('None')

    return cp2k_args, cp2k_OT


def fun_format(s):
    """
    Wrapped a string inside a string for printing purposes
    """
    return '"{}"'.format(s)


# ============================> Distribution <=================================


def distribute_computations(scratch_path, project_name, basisCP2K, potCP2K,
                            cp2k_main, cp2k_guess, path_to_trajectory, blocks,
                            slurm, cwd, nHOMO, couplings_range,
                            algorithm, dt):

    script_name = "script_remote_function.py"
    # Split the trajectory in Chunks and move each chunk to its corresponding
    # directory.
    chunks_trajectory = split_trajectory(path_to_trajectory, blocks, '.')
    chunks_trajectory.sort()
    enumerate_from = 0
    for  i, (file_xyz, l) in enumerate(zip(chunks_trajectory,
                                           string.ascii_lowercase)):
        folder = 'chunk_{}'.format(l)
        os.mkdir(folder)
        shutil.move(file_xyz, folder)

        # HDF5 file where both the Molecular orbitals and coupling are stored
        path_hdf5 = join(scratch_path, '{}.hdf5'.format(folder))
        hamiltonians_dir = join(scratch_path, 'hamiltonians')
        # function to be execute remotely
        work_dir = join(scratch_path, 'batch_{}'.format(i))
        write_python_script(work_dir, folder, file_xyz, project_name,
                            basisCP2K, potCP2K, cp2k_main,
                            cp2k_guess, enumerate_from, script_name,
                            path_hdf5, nHOMO, couplings_range, algorithm, dt)

        # number of geometries per batch
        dim_batch = number_of_geometries(join(folder, file_xyz))
        # Slurm executable
        write_slurm_script(cwd, folder, slurm, script_name, path_hdf5,
                           hamiltonians_dir, (dim_batch * i,
                                              dim_batch * (i + 1) - 3))
        enumerate_from += dim_batch


def write_python_script(
        scratch, folder, file_xyz, project_name, basisCP2K, potCP2K, cp2k_main,
        cp2k_guess, enumerate_from, script_name, path_hdf5, nHOMO,
        couplings_range, algorithm, dt):
    """ Write the python script to compute the PYXAID hamiltonians"""
    path = join(scratch, project_name)
    if not os.path.exists(path):
        os.makedirs(path)
    xs = """
from nac.workflows.workflow_coupling import generate_pyxaid_hamiltonians
from nac.workflows.initialization import initialize
from qmworks import Settings

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
                            path_hdf5=path_hdf5,
                            scratch_path='{}')

# Molecular orbitals calculation input
cp2k_main = Settings()
{}

#Guess Molecular orbitals
cp2k_guess = Settings()
{}

generate_pyxaid_hamiltonians('cp2k', project_name, cp2k_main,
                             guess_args=cp2k_guess,
                             nHOMO={},
                             algorithm='{}',
                             dt={},
                             couplings_range=({},{}),
                             **initial_config)
 """.format(project_name, basisCP2K, potCP2K, path_hdf5, file_xyz, cp2k_main.basis,
            enumerate_from, scratch, dot_print(cp2k_main, parent='cp2k_main'),
            dot_print(cp2k_guess, parent='cp2k_guess'), nHOMO,
            algorithm, dt, *couplings_range)

    with open(join(folder, script_name), 'w') as f:
        f.write(xs)

    return script_name


def write_slurm_script(cwd, folder, slurm, python_script, path_hdf5,
                       hamiltonians, range_batch=None):
    """
    write an Slurm launch script
    """
    python = "python {}\n".format(python_script)
    results_dir = "{}/results_{}".format(cwd, folder)
    mkdir = "mkdir {}\n".format(results_dir)

    # Copy a subset of Hamiltonians
    if range_batch is not None:
        files_hams = '{}/Ham_{{{}..{}}}_*'.format(hamiltonians, *range_batch)
    else:
        files_hams = hamiltonians

    copy = 'cp -r {} {} {}\n'.format(path_hdf5, files_hams, results_dir)
    # Script content
    content = format_slurm_parameters(slurm) + python + mkdir + copy

    # Write the script
    file_name = join(folder, "launch.sh")

    with open(file_name, 'w') as f:
        f.write(content)
    return file_name


def format_slurm_parameters(slurm):
    """
    Format as a string some SLURM parameters
    """
    sbatch = "#SBATCH -{} {}\n".format

    header = "#! /bin/bash\n"
    modules = "\nmodule load cp2k/3.0\nsource activate qmworks\n\n"
    time = sbatch('t', slurm.time)
    nodes = sbatch('N', slurm.nodes)
    tasks = sbatch('n', slurm.tasks)
    name = sbatch('J', slurm.name)

    return ''.join([header, time, nodes, tasks, name, modules])


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


def dot_print(s, parent='s'):
    acc = ''
    for k, v in s.items():
        if not k.startswith('_'):
            if not isinstance(v, Settings):
                acc += '{}.{} = {}\n'.format(parent, k, v)
            else:
                acc += dot_print(v, parent='{}.{}'.format(parent, k))
    return acc


if __name__ == "__main__":
    main()
