#! /usr/bin/env python
"""This program plots the energies of each kohn-sham state along the MD trajectory

Note that you have to provide the location of the folder where the NAMD
hamiltonian elements are stored using the -p flag
"""

from __future__ import annotations

import glob
import argparse
import os
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from nanoqm.analysis import read_energies

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import float64 as f8


def plot_stuff(
    energies: NDArray[f8],
    ts: int,
    ihomo: int,
    nhomos: int,
    nlumos: int,
) -> None:
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


def main(path_hams: str, ts: str, ihomo: int, nhomos: int, nlumos: int) -> None:
    if ts == 'All':
        files = glob.glob(os.path.join(path_hams, 'Ham_*_re'))
        ts_int = len(files)
    else:
        ts_int = int(ts)
    energies = read_energies(path_hams, ts_int)
    plot_stuff(energies, ts_int, ihomo, nhomos, nlumos)


def read_cmd_line(parser: argparse.ArgumentParser) -> tuple[str, str, int, int, int]:
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    return (args.p, args.ts, args.ihomo, args.nhomos, args.nlumos)


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
