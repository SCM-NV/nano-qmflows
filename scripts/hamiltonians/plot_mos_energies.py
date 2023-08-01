#! /usr/bin/env python
"""This program plots the energies of each kohn-sham state along the MD trajectory

Note that you have to provide the location of the folder where the NAMD
hamiltonian elements are stored using the -p flag
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import os
from nanoqm.analysis import read_energies


def plot_stuff(energies, ts, ihomo, nhomos, nlumos):
    """
    energies - a vector of energy values that can be plotted
    """
    dim_x = np.arange(ts)

    plt.xlabel('Time (fs)')
    plt.ylabel('Energy (eV)')
    plt.plot(dim_x, energies[:, ihomo - nhomos: ihomo + nlumos])

    fileName = "MOs_energies.png"
    plt.savefig(fileName, format='png', dpi=300)

    plt.show()


def main(path_hams, ts, ihomo, nhomos, nlumos):
    if ts == 'All':
        files = glob.glob(os.path.join(path_hams, 'Ham_*_re'))
        ts = len(files)
    else:
        ts = int(ts)
    energies = read_energies(path_hams, ts)
    plot_stuff(energies, ts, ihomo, nhomos, nlumos)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['p', 'ts', 'ihomo', 'nhomos', 'nlumos']

    return [getattr(args, p) for p in attributes]


if __name__ == "__main__":

    msg = "plot_decho -p <path/to/hamiltonians> -ts <time window for analysis>\
    -nhomos <number of homo states to be plotted>\
    -nlumos <number of lumo states to be plotted"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        '-p', required=True, help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument(
        '-ts', type=str, default='All', help='Index of the second state')
    parser.add_argument(
        '-ihomo', type=int, required=True, help='Index of the HOMO state')
    parser.add_argument(
        '-nhomos', default=10, type=int, help='Number of homo states to be plotted')
    parser.add_argument(
        '-nlumos', default=10, type=int, help='Number of lumo states to be plotted')

    main(*read_cmd_line(parser))
