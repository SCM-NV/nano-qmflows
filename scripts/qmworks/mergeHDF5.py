#! /usr/bin/env python
from os.path import join
import argparse
import h5py

"""
This program merges the HDF5 files obtained from the SCF calculations when
the MD trajectory has been split in more than one block.
Example:

mergeHDF5.py -i chunk_a.hdf5 chunk_b.hdf5 chunk_c.hdf5 -o total.hdf5

An empty total.hdf5 file should be already available before using the script.
"""

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
    """
    Merge Recursively two hdf5 Files
    """
    with h5py.File(inp, 'r') as f5, h5py.File(out, 'r+') as g5:
        keys = f5.keys()
        for k in keys:
            if k not in g5:
                g5.create_group(k)
            keys2 = f5[k].keys()
            for l in keys2:
                if l not in g5[k]:
                    path = join(k, l)
                    print("Copying pAth: ", path)
                    f5.copy(path, g5[k])


def main():
    inps, out = read_cmd_line()
    for i in inps:
        mergeHDF5(i, out)


if __name__ == "__main__":
    main()
