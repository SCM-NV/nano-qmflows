__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from functools import reduce, partial
from os.path import join

import h5py
import os
import shutil
import subprocess

# ==================> Internal modules <==========


# ==============================> Main <==================================


def split_trajectory(path, nBlocks, pathOut):
    """
    Split an XYZ trajectory in n Block and write
    them in a given path.

    :Param path: Path to the XYZ file.
    :type path: String
    :param nBlocks: number of Block into which the xyz file is split.
    :type nBlocks: Int
    :param pathOut: Path were the block are written.
    :type pathOut: String
    :returns: tuple (Number of structures per block, path to blocks)
    """
    with open(path, 'r') as f:
            l = f.readline()  # Read First line
            numat = int(l.split()[0])

    # Number of lines in the file
    cmd = "wc -l {}".format(path)
    l = subprocess.check_output(cmd.split()).decode()
    lines = int(l.split()[0])
    # Number of points in the xyz file
    nPoints = lines // (numat + 2)
    # Number of points for each chunk
    nChunks = nPoints // nBlocks
    # Number of lines per block
    lines_per_block = nChunks * (numat + 2)
    # Path where the splitted xyz files are written
    prefix = join(pathOut, 'chunk_xyz_')
    cmd = 'split -a 1 -l {} {} {}'.format(lines_per_block, path, prefix)
    subprocess.run(cmd, shell=True)

    # the unix split command starts naming from 'a' the files created after
    # splitting
    lim = ord('a')
    folders = [prefix + chr(x) for x in range(lim, lim + nChunks)]

    return nChunks, folders


def process_arguments(xs):
    """
    Initialize and submit the slurm scripts to compute the derivative coupling.

    :param xs: String containing the input arguments.
    """
    def make_dictionary(acc, xs):
        """ accumulate key, value pairs in a dictionary"""
        key, val = xs
        key = key.strip()
        acc[key] = make_proper_val(val)
        return acc

    def make_proper_val(xs):
        """remove blanks spaces and trailing comments """
        if '"' in xs:
            return xs.split('"')[1]
        else:
            # remove comments
            return xs.split('#')[0]
    
    # Create a dictionary with the arguments provided by the user
    ls = filter(lambda x: not x.startswith('#'), xs.splitlines())
    xss = [x for x in [s.split('=') for s in ls] if len(x) >= 2]
    ds = reduce(make_dictionary, xss, {})

    project_name = ds['Project_name']
    package_name = ds['QM_package']
    scratch = ds['Scratch_folder']

    # Work_dir
    scratch_path = join(scratch, project_name)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch)
    
    # HDF5 path
    path_hdf5 = join(scratch_path, 'quantum.hdf5')

    # number of block to run simultaneusly
    nBlocks = int(ds['NumberOfTrajectoryBlocks'].strip())

    # Path to the MD geometries
    path_traj_xyz = ds['Trajectory_path']
    
    # Split the md xyz file into smaller folders
    nChunks, paths_xyz = split_trajectory(path_traj_xyz, nBlocks, scratch_path)

    # Cp2k arguments
    basis_folder = ds['Basis_set_folder']
    potential_folder = ds['Potential_folder']
    basis = ds['Basis_set']
    potential = ds['Potential']
    added_mos = ds['Added_MOs']
    cell_parameters = ds['Cell_parameters']

    # Queue system configuration
    nodes = ds['NumberOfNodesPerBlock']
    tasks_per_node = int(ds['NumberOfProcsPerBlock'])

    # string containing the scripts to run remotelly
    fun_scripts = partial(generate_python_script, package_name, project_name,
                          path_hdf5, basis_folder, potential_folder, basis,
                          potential, cell_parameters, added_mos)

    cwd = os.path.realpath(".")
    scripts = ['workflow_coupling.py']
    for ps, i in zip(paths_xyz, range(nBlocks)):
        work_dir = join(cwd, 'block_{}'.format(i))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        copy_scripts(cwd, work_dir, scripts)
        script_name = join(work_dir, 'script_block_{}.py'.format(i))
        fun_scripts(ps, nChunks * i, script_name)
        write_bash_script(work_dir, script_name, nodes=nodes,
                          tasks_per_node=tasks_per_node,
                          load_cp2k='cp2k/2.5.1')
    print("Bash scripts to submit the Nonadiabatic coupling were generated!!")


def copy_scripts(src_dir, dest_dir, names):
    """
    Copy the scripts to the given dir.
    """
    for f in names:
        src = join(src_dir, f)
        dest = join(dest_dir, f)
        if not os.path.exists(dest):
            shutil.copyfile(src, dest)


def write_bash_script(work_dir, script, system='slurm', walltime='48:00:00',
                      **kwargs):
    """
    Generate a SLURM script to submit a chunk of the trajectory.
    :param work_dir: Path where the script is executed.
    :type work_dir: String
    :param script: Name of the Python script to be execute remotelly.
    :type script: String
    """
    def call_slurm(nodes=1, tasks_per_node=1, load_cp2k='cp2k'):

        inp = ''
        inp += '#! /bin/bash\n'
        inp += '#SBATCH -t {}\n'.format(walltime)
        inp += '#SBATCH --nodes={}\n'.format(int(nodes))
        inp += '#SBATCH --ntasks-per-node={}\n\n'.format(int(tasks_per_node))
        inp += 'module load {}\n'.format(load_cp2k)
        inp += 'source activate qmworks\n\n'
        inp += 'python {}'.format(script)
        return inp
        
    d = {'slurm': call_slurm}
    inp = d[system](**kwargs)

    path = join(work_dir, 'launchSlurm.sh')

    with open(path, 'w') as f:
        f.write(inp)


def generate_python_script(package_name, project_name, path_hdf5, basis_folder,
                           potential_folder, basis, potential, cell_parameters,
                           added_mos, path_chunk_xyz, enumerate_from,
                           script_name):
    """
    creates a python script to be execute remotely using sbatch
    """
    ind = '    '
    inp = ''
    inp += 'from qmworks import Settings\n'
    inp += ('from workflow_coupling import (generate_pyxaid_hamiltonians,'
            ' split_file_geometries)\n')
    inp += 'import plams\n\n'
    inp += 'def main():\n'
    inp += '{}plams.init()'.format(ind)
    inp += '{}# create Settings for the Cp2K Jobs\n\n'.format(ind)
    inp += '{}cp2k_args = Settings()\n'.format(ind)
    inp += '{}cp2k_args.basis = "{}"\n'.format(ind, basis)
    inp += '{}cp2k_args.potential = "{}"\n'.format(ind, potential)
    inp += '{}cp2k_args.cell_parameters = {}\n'.format(ind, cell_parameters)
    inp += ('{}cp2k_args.specific.cp2k.force_eval.'
            'dft.scf.added_mos = {}\n\n'.format(ind, added_mos))
    inp += ('{}geometries = '
            'split_file_geometries("{}")\n\n'.format(ind, path_chunk_xyz))
    inp += ('{}generate_pyxaid_hamiltonians("{}", "{}",'
            ' geometries, cp2k_args, path_hdf5="{}",'
            ' enumerate_from={})\n\n'.format(ind, package_name.lower(),
                                             project_name, path_hdf5,
                                             enumerate_from))
    inp += '{}plams.finish()'.format(ind)
    
    inp += '\n\nif __name__ == "__main__":\n{}main()'.format(ind)

    with open(script_name, 'w') as f:
        f.write(inp)


def generate_hdf5_file(project_name, scratch_folder):
    """
    Generates a unique path to store temporal data as HDF5
    """
    scratch = join(scratch_folder, project_name)
    if not os.path.exists(scratch):
        os.makedirs(scratch)

    return join(scratch, 'quantum.hdf5')



