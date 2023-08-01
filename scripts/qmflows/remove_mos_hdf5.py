#! /usr/bin/env python

from __future__ import annotations

from os.path import join
import argparse
import h5py
import numpy as np

from nanoqm import logger


def main(
    project_name: str,
    path_hdf5: str,
    indices: list[int],
    overlap_flag: bool,
    mo_flag: bool,
) -> None:

    # Shift indices to start from 0
    indices_ar = np.array(indices) - 1

    # path to the failed points
    if mo_flag:
        mos = [join(project_name, f'point_{i}') for i in indices_ar]
    else:
        mos = []
    if overlap_flag:
        overlaps = [join(project_name, f'overlaps_{i}') for i in indices_ar]
    else:
        overlaps = []

    # Concatenate both Molecular orbitals and Overlaps
    paths = mos + overlaps

    with h5py.File(path_hdf5, 'r+') as f5:
        for p in paths:
            if p in f5:
                logger.info("removing: ", p)
                del f5[p]


def read_cmd_line(parser) -> tuple[str, str, list[int], bool, bool]:
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    return (args.pn, args.hdf5, args.i, args.o, args.mo)


if __name__ == "__main__":

    msg = " remove_couplings -pn <ProjectName> -hdf5 <path/to/hdf5 file> -o False"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-pn', required=True,
                        help='project name')
    parser.add_argument('-hdf5', required=True,
                        help='Path to the HDF5')
    parser.add_argument(
        '-o', help='flag to remove the overlaps', action='store_true')
    parser.add_argument('-mo', help='flag to remove the molecular overlaps',
                        action='store_true')
    parser.add_argument('-i', help='Indices of the Molecular orbitals', required=True,
                        nargs='+', type=int)
    main(*read_cmd_line(parser))
