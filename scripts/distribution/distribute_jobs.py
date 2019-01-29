#!/usr/bin/env python

from nac.workflows.input_validation import process_input
from nac.workflows.initialization import split_trajectory
from os.path import join
from qmflows.utils import settings2Dict

import argparse
import os
import shutil
import subprocess
import yaml


def read_cmd_line():
    """
    Read the input file and the workflow name from the command line
    """
    msg = "distribute_jobs.py -i input.yml"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-i', required=True,
                        help="Input file in YAML format")

    args = parser.parse_args()
    return args.i


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
    # command line argument
    input_file = read_cmd_line()

    # Read and process input
    dict_input = process_input(input_file, "distribute_derivative_couplings")

    # Write scripts to run calculations
    distribute_computations(dict_input)


def distribute_computations(dict_input: dict) -> None:
    """
    Prepare the computation
    """
    config = dict_input['general_settings']
    workdir = dict_input["workdir"]
    scratch_path = config["scratch_path"]
    enumerate_from = config["enumerate_from"]
    job_scheduler = dict_input["job_scheduler"]

    # Check if workdir exits otherwise create it
    os.makedirs(workdir, exist_ok=True)

    # Split the trajectory in Chunks and move each chunk to its corresponding
    # directory.
    chunks_trajectory = split_trajectory(
        config["path_traj_xyz"], dict_input["blocks"], workdir)
    chunks_trajectory.sort()

    for index, file_xyz in enumerate(chunks_trajectory):
        folder_path = join(workdir, 'chunk_{}'.format(index))

        # Move xyz to temporal file
        os.makedirs(folder_path, exist_ok=True)
        shutil.move(file_xyz, folder_path)

        # Scratch directory
        batch_dir = join(scratch_path, 'batch_{}'.format(index))
        os.makedirs(batch_dir, exist_ok=True)

        # HDF5 file where both the Molecular orbitals and coupling are stored
        path_hdf5 = join(scratch_path, 'chunk_{}.hdf5'.format(index))

        # Change hdf5 and trajectory path of each batch
        config["path_traj_xyz"] = file_xyz
        config["path_hdf5"] = path_hdf5

        # files with PYXAID
        hamiltonians_dir = join(scratch_path, 'hamiltonians')

        # number of geometries per batch
        dim_batch = compute_number_of_geometries(join(folder_path, file_xyz))

        # Edit the number of geometries to compute
        config["enumerate_from"] = enumerate_from
        # Write input file
        write_input(folder_path, config)

        # Slurm executable
        if job_scheduler["scheduler"].upper() == "SLURM":
            write_slurm_script(
                folder_path, index, job_scheduler, path_hdf5, hamiltonians_dir, dim_batch)
        else:
            msg = "The request job_scheduler: {} It is not implemented".formant()
            raise RuntimeError(msg)

        # change the window of molecules to compute
        enumerate_from += dim_batch


def write_input(folder_path: str, config: dict) -> None:
    """ Write the python script to compute the PYXAID hamiltonians"""
    file_path = join(folder_path, "input.yml")

    # transform settings to standard dictionary
    config["settings_main"] = settings2Dict(config["settings_main"])
    config["settings_guess"] = settings2Dict(config["settings_guess"])

    # basis and potential
    config["path_basis"] = os.path.abspath(config["path_basis"])
    config["path_potential"] = os.path.abspath(config["path_potential"])

    d = {"workflow": "derivative_couplings", "general_settings": config}

    with open(file_path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, allow_unicode=True)


def write_slurm_script(
        folder_path: str, index: int, slurm_config: dict, path_hdf5: str,
        hamiltonians_dir: str, dim_batch: int):
    """
    write an Slurm launch script
    """
    python = "\n\nrun_workflow.py -i input.yml\n"
    results_dir = join(folder_path, "results_chunk_" + str(index))
    mkdir = "\nmkdir {}\n".format(results_dir)

    # Copy a subset of Hamiltonians
    range_batch = (dim_batch * index, dim_batch * (index + 1) - 3)
    files_hams = '{}/Ham_{{{}..{}}}_*'.format(hamiltonians_dir, *range_batch)

    copy = 'cp -r {} {} {}\n'.format(path_hdf5, files_hams, results_dir)
    # Script content
    content = format_slurm_parameters(slurm_config) + python + mkdir + copy

    # Write the script
    with open(join(folder_path, "launch.sh"), 'w') as f:
        f.write(content)


def format_slurm_parameters(slurm):
    """
    Format as a string some SLURM parameters
    """
    sbatch = "#SBATCH -{} {}\n".format

    header = "#! /bin/bash\n"
    time = sbatch('t', slurm["wall_time"])
    nodes = sbatch('N', slurm["nodes"])
    tasks = sbatch('n', slurm["tasks"])
    name = sbatch('J',  slurm["job_name"])

    modules = slurm["load_modules"]

    return ''.join([header, time, nodes, tasks, name, modules])


def compute_number_of_geometries(file_name):
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
