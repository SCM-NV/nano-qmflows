#! /usr/bin/env python
"""This programs plots the electronic coupling between two states.

  It reads all Ham_*_im files and cache them in a tensor saved on disk.
  Usage:
  plot_couplings.py -p . -s1 XX -s2 YY -dt 1.0

p = path to the hamiltonian files
s1 = state 1 index
s2 = state 2 index
dt = time step in fs
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os.path

r2meV = 13605.698  # From Rydeberg to eV


def main(path_output: str, s1: int, s2: int, dt: float) -> None:
    # Check if the file with couplings exists
    if not os.path.isfile('couplings.npy'):
        # Check all the files stored
        files_im = glob.glob('Ham_*_im')
        # Read the couplings
        couplings = np.stack(
           [np.loadtxt(f'Ham_{f}_im') for f in range(len(files_im))]
        )
        # Save the file for fast reading afterwards
        np.save('couplings', couplings)
    else:
        couplings = np.load('couplings.npy')
        ts = np.arange(couplings.shape[0]) * dt
        plt.plot(ts, couplings[:, s1, s2] * r2meV)
        plt.xlabel('Time (fs)')
        plt.ylabel('Energy (meV)')
        plt.show()


def read_cmd_line(parser: argparse.ArgumentParser) -> tuple[str, int, int, float]:
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    return (args.p, args.s1, args.s2, args.dt)


if __name__ == "__main__":
    msg = "plot_decho -p <path/to/hamiltonians> -s1 <State 1> -s2 <State 2>\
    -dt <time step>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True,
                        help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-s1', required=True, type=int,
                        help='Index of the first state')
    parser.add_argument('-s2', required=True, type=int,
                        help='Index of the second state')
    parser.add_argument('-dt', type=float, default=1.0,
                        help='Index of the second state')
    main(*read_cmd_line(parser))
