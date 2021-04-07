#! /usr/bin/env python
import argparse
from os.path import join

import h5py


def main(path_hdf5: str, remove_overlaps: bool):
    """Remove unused array from the HDF5."""
    path_swaps = ['swaps']
    paths_overlaps_corrected = [
        join(f'overlaps_{i}/mtx_sji_t0_corrected') for i in range(10000)]
    if remove_overlaps:
        paths_overlaps = [
            join(f'overlaps_{i}/mtx_sji_t0') for i in range(10000)]
    else:
        paths_overlaps = []

    with h5py.File(path_hdf5, 'r+') as f5:
        paths_css = list(filter(lambda x: 'coupling' in x, f5.keys()))
        paths = paths_css + paths_overlaps_corrected + path_swaps + paths_overlaps
        for p in (p for p in paths if p in f5):
            print("removing: ", p)
            del f5[p]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("removeHDF5folders.py")
    parser.add_argument('-hdf5', required=True,
                        help='Path to the HDF5 file')
    parser.add_argument(
        '-o', help='Remove the overlap matrices', action='store_true')
    args = parser.parse_args()
    main(args.hdf5, args.o)
