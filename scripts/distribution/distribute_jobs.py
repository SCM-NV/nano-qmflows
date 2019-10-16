#!/usr/bin/env python

from nac.common import (DictConfig, read_cell_parameters_as_array)
from nac.workflows.input_validation import process_input
from nac.workflows.initialization import split_trajectory
from os.path import join
from qmflows.utils import settings2Dict

import argparse
import numpy as np
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

    with open(input_file, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Read and process input
    workflow_type = args['workflow'].lower()
    print("calling process_input: ", workflow_type)
    dict_input = process_input(input_file, workflow_type)

    # Write scripts to run calculations
    if workflow_type == "distribute_derivative_couplings":
        distribute_computations(dict_input, hamiltonians=True)
    else:
        distribute_computations(dict_input)


def distribute_computations(config: dict, hamiltonians=False) -> None:
    """
    Prepare the computation and write the scripts
    """
    # Check if workdir exits otherwise create it
    os.makedirs(config.workdir, exist_ok=True)

    # Split the trajectory in Chunks and move each chunk to its corresponding
    # directory.
    chunks_trajectory = split_trajectory(
        config.path_traj_xyz, config.blocks, config.workdir)
    chunks_trajectory.sort()

    file_cell_parameters = config.cp2k_general_settings.get("file_cell_parameters")
    if file_cell_parameters is not None:
        header_array_cell_parameters = read_cell_parameters_as_array(file_cell_parameters)

    for index, file_xyz in enumerate(chunks_trajectory):
        folder_path = os.path.abspath(join(config.workdir, f'chunk_{index}'))

        dict_input = DictConfig({
            'folder_path': folder_path, "file_xyz": file_xyz, 'index': index})

        create_folders(config, dict_input)
        # HDF5 file where both the Molecular orbitals and coupling are stored
        config.path_hdf5 = join(config.scratch_path, f'chunk_{index}.hdf5')

        # Change hdf5 and trajectory path of each batch
        config["path_traj_xyz"] = file_xyz

        if file_cell_parameters is not None:
            add_chunk_cell_parameters(header_array_cell_parameters, config, dict_input)

        # files with PYXAID
        if hamiltonians:
            dict_input.hamiltonians_dir = join(config.scratch_path, 'hamiltonians')

        # number of geometries per batch
        dict_input.dim_batch = compute_number_of_geometries(join(folder_path, file_xyz))

        # Write input file
        write_input(folder_path, config)

        # Slurm executable
        if config.job_scheduler["scheduler"].upper() == "SLURM":
            write_slurm_script(config, dict_input)
        else:
            msg = "The request job_scheduler: {} It is not implemented".formant()
            raise RuntimeError(msg)

        # change the window of molecules to compute
        config['enumerate_from'] += dict_input.dim_batch


def write_input(folder_path: str, config: dict) -> None:
    """ Write the python script to compute the PYXAID hamiltonians"""
    file_path = join(folder_path, "input.yml")

    # transform settings to standard dictionary
    config = settings2Dict(config)
    config["cp2k_general_settings"]["cp2k_settings_main"] = settings2Dict(
        config["cp2k_general_settings"]["cp2k_settings_main"])
    config["cp2k_general_settings"]["cp2k_settings_guess"] = settings2Dict(
        config["cp2k_general_settings"]["cp2k_settings_guess"])

    # basis and potential
    config["cp2k_general_settings"]["path_basis"] = os.path.abspath(
        config["cp2k_general_settings"]["path_basis"])

    # remove keys from input
    for k in ['blocks', 'job_scheduler', 'mo_index_range',
              'workdir']:
        del config[k]

    # rename the workflow to execute
    workflow_type = config["workflow"].lower()
    if workflow_type == "distribute_derivative_couplings":
        config['workflow'] = "derivative_couplings"
    elif workflow_type == "distribute_absorption_spectrum":
        config['workflow'] = "absorption_spectrum"
    with open(file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def create_folders(config: dict, dict_input: dict):
    """
    Create folder for each batch and copy the xyz
    """
    # Move xyz to temporal file
    os.makedirs(dict_input.folder_path, exist_ok=True)
    shutil.move(dict_input.file_xyz, dict_input.folder_path)

    # Scratch directory
    batch_dir = join(config.scratch_path, f'batch_{dict_input.index}')
    os.makedirs(batch_dir, exist_ok=True)


def write_slurm_script(config: dict, dict_input: dict):
    """
    write an Slurm launch script
    """
    index = dict_input.index
    python = "\n\nrun_workflow.py -i input.yml\n"
    results_dir = "results_chunk_" + str(index)
    mkdir = "\nmkdir {results_dir}\n"
    slurm_config = config.job_scheduler

    # Copy a subset of Hamiltonians
    if dict_input.get("hamiltonians_dir") is None:
        copy = ""
    else:
        dim_batch = dict_input.dim_batch
        range_batch = (dim_batch * index, dim_batch * (index + 1) - 3)
        files_hams = '{}/Ham_{{{}..{}}}_*'.format(
            dict_input.hamiltonians_dir, *range_batch)
        copy = f'cp -r {config.path_hdf5} {files_hams} {results_dir}\n'

    # Script content
    content = format_slurm_parameters(slurm_config) + python + mkdir + copy

    # Write the script
    with open(join(dict_input.folder_path, "launch.sh"), 'w') as f:
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
    queue = sbatch('p', slurm["queue_name"])

    modules = slurm["load_modules"]

    return ''.join([header, time, nodes, tasks, name, queue, modules])


def compute_number_of_geometries(file_name):
    """
    Count the number of geometries in XYZ formant in a given file.
    """

    with open(file_name, 'r') as f:
        numat = int(f.readline())

    cmd = f"wc -l {file_name}"
    wc = subprocess.getoutput(cmd).split()[0]

    lines_per_geometry = numat + 2

    return int(wc) // lines_per_geometry


def add_chunk_cell_parameters(
        header_array_cell_parameters: tuple, config: dict, dict_input: dict) -> None:
    """
    Add the corresponding set of cell parameters for a given chunk
    """
    path_file_cell_parameters = join(dict_input.folder_path, "cell_parameters.txt")

    # Adjust settings
    config.cp2k_general_settings["file_cell_parameters"] = path_file_cell_parameters

    # extract cell parameters for a chunk
    header, arr = header_array_cell_parameters
    size = int(np.ceil(arr.shape[0] / config.blocks))
    low = dict_input.index * size
    high = low + size
    step = config.stride
    if high < arr.shape[0]:
        matrix_cell_parameters = arr[low:high:step, :]
    else:
        matrix_cell_parameters = arr[low::step, :]

    # adjust indices of the cell parameters
    size = matrix_cell_parameters.shape[0]
    matrix_cell_parameters[:, 0] = np.arange(size)
    # Save file
    fmt = '%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f'
    np.savetxt(path_file_cell_parameters, matrix_cell_parameters, header=header, fmt=fmt)


if __name__ == "__main__":
    main()
