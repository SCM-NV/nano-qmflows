#! /usr/bin/env python
"""This program reads the ouput files, out and me_pop,from a NAMD simulation run with pyxaid.

It average the populations of each state upon several initial conditions and
also allows to define macrostates.
A macrostate is formed by a group of (micro)states indexed according to pyxaid numbering.
For example, if you have donor-acceptor system, you can group all (micro)states formed by
excited states localized within the donor/acceptor and or charge transfer states
into a few macrostates. This helps the analysis of the population traces when a large
number of excited states is included in the NAMD simulation.

Example:

 plot_states_pops.py -p . -ms "[ [0], [1,3], [4,6] ]" -nconds 6

This means that the path with the output files is the current path '.' .
The number of initial conditions over which the populations are averaged are 6.
The macrostates are defined as a list of lists. The [0] is the ground state, [1,3]
form a macrostate of states 1 and 3 indexed as in pyxaid, [4,6] forms
a macrostates of states 4 and 6.
"""

from __future__ import annotations

import os
import argparse
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from nanoqm.analysis import parse_list_of_lists

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import float64 as f8


def plot_stuff(outs: NDArray[f8], pops: NDArray[f8]) -> None:
    """energies - a vector of energy values that can be plotted."""
    dim_x = np.arange(outs.shape[0])

    ax1 = plt.subplot(121)
    ax1.set_title('SH Population ')
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Population')
    ax1.plot(dim_x, outs[0:, :])

    ax2 = plt.subplot(122)
    ax2.set_title('SE Population ')
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Population')
    ax2.plot(dim_x, pops[0:, :])

    fileName = "State Population.png"
    plt.savefig(fileName, format='png', dpi=300)

    plt.show()


def read_populations(path: str, fn: str, nconds: int, ms: list[list[int]]) -> list[NDArray[f8]]:
    inpfile = os.path.join(path, fn)
    cols = list(map(lambda row: tuple(map(lambda x: x * 2 + 3, row)), ms))
    xs = [
        np.stack(
            [np.loadtxt(f'{inpfile}{j}', usecols=col, dtype=np.float64) for j in range(nconds)]
        ) for col in cols
    ]
    return xs


def main(path_output: str, ms: list[list[int]], nconds: int) -> None:

    outs = read_populations(path_output, 'out', nconds, ms)
    pops = read_populations(path_output, 'me_pop', nconds, ms)

    outs_avg = [np.average(out, axis=0) for out in outs]
    pops_avg = [np.average(pop, axis=0) for pop in pops]

    outs_fin, pops_fin = [], []

    for x in outs_avg:
        if x.ndim == 1:
            outs_fin.append(x)
        else:
            outs_fin.append(np.sum(x, axis=1))
    outs_fin_ar = np.array(outs_fin).T

    for x in pops_avg:
        if x.ndim == 1:
            pops_fin.append(x)
        else:
            pops_fin.append(np.sum(x, axis=1))
    pops_fin_ar = np.array(pops_fin).T

    plot_stuff(outs_fin_ar, pops_fin_ar)


def read_cmd_line(parser) -> tuple[str, str, int]:
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    return (args.p, args.ms, args.nconds)


# ============<>===============
if __name__ == "__main__":
    msg = "plot_states_pops -p <path/to/output>\
    -ms <list of microstates state indexes>\
    -nconds <number of initial conditions>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True,
                        help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-ms', type=str, required=True,
                        help='Macrostate defined as a list of microstates')
    parser.add_argument('-nconds', type=int, required=True,
                        help='Number of initial conditions')

    p, ms, nconds = read_cmd_line(parser)
    main(p, parse_list_of_lists(ms), nconds)
