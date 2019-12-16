#! /usr/bin/env python
import argparse
from os.path import join

import h5py


def main(project_name, path_hdf5, remove_overlaps):
    """Remove unused array from the HDF5."""
    path_swaps = [join(project_name, 'swaps')]
    paths_overlaps_corrected = [
        join(project_name, f'overlaps_{i}/mtx_sji_t0_corrected') for i in range(10000)]
    if remove_overlaps:
        paths_overlaps = [
            join(project_name, 'overlaps_{i}/mtx_sji_t0') for i in range(10000)]
    else:
        paths_overlaps = []

    with h5py.File(path_hdf5, 'r+') as f5:
        xs = filter(lambda x: 'coupling' in x, f5[project_name].keys())
        paths_css = [join(project_name, x) for x in xs]
        paths = paths_css + paths_overlaps_corrected + path_swaps + paths_overlaps
        for p in (p for p in paths if p in f5):
            print("removing: ", p)
            del f5[p]


def read_cmd_line(parser):
    """Parse Command line options."""
    args = parser.parse_args()

    attributes = ['pn', 'hdf5', 'o']

    return [getattr(args, p) for p in attributes]


if __name__ == "__main__":

    msg = " remove_couplings -pn <ProjectName> -hdf5 <path/to/hdf5 file> -o False"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-pn', required=True,
                        help='project name')
    parser.add_argument('-hdf5', required=True,
                        help='Index of the first state')
    parser.add_argument(
        '-o', help='Remove the overlap matrices', action='store_true')
    main(*read_cmd_line(parser))
