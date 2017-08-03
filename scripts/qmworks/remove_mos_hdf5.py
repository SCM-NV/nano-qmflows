#! /usr/bin/env python

from os.path import join
import argparse
import h5py
import numpy as np


def main(project_name, path_hdf5, indices):

    # Shift indices to start from 0
    indices = np.array(indices) - 1

    # path to the failed points
    root_paths = join(project_name, 'point_{}')
    root_overlaps = join(project_name, 'overlaps_{}')
    mos = [root_paths.format(i) for i in indices]
    overlaps = [root_overlaps.format(i) for i in indices]

    # Concatenate both Molecular orbitals and Overlaps
    paths = mos + overlaps

    with h5py.File(path_hdf5, 'r+') as f5:
        for p in paths:
            if p in f5:
                print("removing: ", p)
                del f5[p]


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['pn', 'hdf5', 'i']

    return [getattr(args, p) for p in attributes]


if __name__ == "__main__":

    msg = " remove_couplings -pn <ProjectName> -hdf5 <path/to/hdf5 file> -o False"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-pn', required=True,
                        help='project name')
    parser.add_argument('-hdf5', required=True,
                        help='Index of the first state')
    parser.add_argument('-i', help='Indices of the Molecular orbitals', required=True,
                        nargs='+', type=int)
    main(*read_cmd_line(parser))
