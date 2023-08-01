#! /usr/bin/env python

"""
This program merges the HDF5 files obtained from the SCF calculations when
the MD trajectory has been split in more than one block.
Example:

mergeHDF5.py -i chunk_a.hdf5 chunk_b.hdf5 chunk_c.hdf5 -o total.hdf5

An empty total.hdf5 file should be already available before using the script.
"""
import argparse
import h5py
import os

# ====================================<>=======================================
msg = " script -i <Path(s)/to/source/hdf5> -o <path/to/destiny/hdf5>"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-i', required=True,
                    help='Path(s) to the HDF5 to merge', nargs='+')
parser.add_argument('-o', required=True,
                    help='Path to the HDF5 were the merge is going to be stored')


def read_cmd_line():
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    inp = args.i
    out = args.o

    return inp, out
# ===============><==================


def mergeHDF5(inp, out):
    """Merge Recursively two hdf5 Files."""
    with h5py.File(inp, 'r') as f5, h5py.File(out, 'r+') as g5:
        merge_recursively(f5, g5)


def merge_recursively(f, g):
    """Traverse all the groups tree and copy the different datasets."""
    for k in f.keys():
        if k not in g:
            if isinstance(f[k], h5py.Dataset):
                f.copy(k, g)
            else:
                g.create_group(k)
                merge_recursively(f[k], g[k])
        elif isinstance(f[k], h5py.Group):
            merge_recursively(f[k], g[k])


def main():
    inps, out = read_cmd_line()
    if not os.path.exists(out):
        touch(out)
    for i in inps:
        mergeHDF5(i, out)


def touch(fname, times=None):
    """Equivalent to unix touch command"""
    with open(fname, 'a'):
        os.utime(fname, times)


if __name__ == "__main__":
    main()
