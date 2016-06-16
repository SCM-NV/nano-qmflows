__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <=========================
from os.path import join

import argparse
import h5py
import getpass
import os

# ========================> Command line args  <===============================
msg = " script -d out"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('--name', required=True, help='Project\'s name')


def read_cmd_line():
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    name = args.name

    return name
# ===============================> Main  <=====================================


def main():
    """
    Read from an HDF5 file what Jobs are already finished
    """
    
    project_name = read_cmd_line()

    # User variables
    username = getpass.getuser()
    
    # Work_dir
    scratch = "/scratch-shared"
    scratch_path = join(scratch, username, project_name)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # HDF5 path
    path_hdf5 = join(scratch_path, 'quantum.hdf5')

    with h5py.File(path_hdf5, 'r') as f5:
        dset = f5[project_name]
        ks = list(dset.keys())

    print("The following Jobs have been completed:\n")
    print(ks)

    
if __name__ == "__main__":
    main()

