
from distribute_jobs import (SLURM, format_slurm_parameters, write_python_script)
from os.path import join
from qmworks import Settings

import fnmatch
import h5py
import os


def main():
    # ========== Fill in the following variables
    # Varaible to define the Path ehere the Cp2K jobs will be computed
    scratch = "/scratch-shared/fza900/Cd33Se33"
    project_name = 'Cd33Se33'  # name use to create folders

    # Path to the basis set used by Cp2k
    basisCP2K = "/home/fza900/Cp2k/cp2k_basis/BASIS_MOLOPT"
    potCP2K = "/home/fza900/Cp2k/cp2k_basis/GTH_POTENTIALS"

    path_to_trajectory = "eightpoints.xyz"

    # Basis
    basis = "DZVP-MOLOPT-SR-GTH"
    # ============== End of User definitions ===================================

    cp2k_args = Settings()
    cp2k_args.basis = basis

    # Current Work Directory
    cwd = os.getcwd()

    # Results folder
    results_dir = join(cwd, 'total_results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Merge all the HDF5 files        
    file_hdf5 = merge_hdf5(scratch, project_name, cwd ,results_dir)

    # compute missing couplings
    script_name = "merge_data.py"
    
    write_python_script(scratch, 'total_results', path_to_trajectory, project_name,
                        basisCP2K, potCP2K, cp2k_args, Settings(), 0, script_name,
                        file_hdf5)

    # Script using SLURM
    write_slurm_script(results_dir, script_name)

    
    
def write_slurm_script(scratch, results_dir):
    slurm = SLURM(1, 24, "00:60:00", "merged_namd")
    
    python = "python {}\n".format(results_dir)
    copy = "cp -r {} {}".format(results_dir, join(scratch, 'hamiltonians'))

    return format_slurm_parameters(slurm) + python + copy
    
def merge_hdf5(scratch, project_name, cwd, results_dir):
    """
    Merge all the hdf5 into a unique file.
    """
    # create path to hdf5 containing all the results
    file_hdf5 = join(results_dir, '{}.hdf5'.format(project_name))

    # Equivalent to touch in unix
    with open(file_hdf5, 'a'):
        os.utime(file_hdf5)

    # read all the HDF5 of the project
    files = fnmatch.filter(os.listdir(scratch), '*.hdf5')

    print("files: ", files)
    # Merge the files into one
    for f in files:
        path = join(scratch, f)
        print("Merging file: ", path)
        merge_files(path, file_hdf5)

    return file_hdf5

def merge_files(file_inp, file_out):
    """
    Merge Recursively two hdf5 Files
    """
    print("Merging files: ", file_inp, file_out)
    with h5py.File(file_inp, 'r') as f5, h5py.File(file_out, 'r+') as g5:
        for k in f5.keys():
            if k not in g5:
                g5.create_group(k)
            for l in f5[k].keys():
                if l not in g5[k]:
                    path = join(k, l)
                    f5.copy(path, g5[k])


if __name__ == "__main__":
    main()
