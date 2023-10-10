#! /usr/bin/env python

"""
This program finds the indexes of the states for electron and hole cooling calculations (ONLY!) \
extrapolated at a desired initial condition. It reads the me_energies0 file from \
pyxaid [at initial condition t=0].

Example:

 iconds_excess_energy.py -p . -nstates 30 -iconds 0 100 200 300 -excess 0.5 -delta 0.2

n_states is the total number of states from the pyxaid simulation.
iconds is a list of initial conditions for the desired simulations
excess is an excess of energy (in eV) from where to begin the electron/hole cooling
delta is an energy range to select states around the excess energy

"""
from __future__ import annotations

import os
import argparse
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import float64 as f8


def read_energies(path: str, fn: str, nstates: int) -> NDArray[f8]:
    inpfile = os.path.join(path, fn)
    cols = tuple(range(5, nstates * 2 + 5, 2))
    xs = np.loadtxt(f'{inpfile}', usecols=cols)
    return xs


def main(path_output: str, nstates: int, iconds: list[int], excess: float, delta: float) -> None:

    # Read Energies
    energies = read_energies(path_output, 'me_energies0', nstates)

    # HOMO-LUMO gap at each time t
    lowest_hl_gap = np.amin(energies[:, 1:], axis=1)
    lowest_hl_gap = lowest_hl_gap.reshape(lowest_hl_gap.shape[0], 1)

    # Scale the energies to calculate the excess energies over the CB and VB
    en_scaled = energies[:, 1:] - lowest_hl_gap

    # Find the index of the states with a given excess energy
    indexes = [np.where(
        (en_scaled[iconds[i]] > excess-delta) & (en_scaled[iconds[i]] < excess + delta))
        for i in range(len(iconds))]

    # Print the states
    t = 'Time Init Cond    List with State Indexes\n'
    for i in range(len(iconds)):
        t += f' {iconds[i]}           {indexes[i][0] + 1}\n'

    with open('initial_conditions.out', 'w', encoding="utf8") as f:
        f.write(t)


def read_cmd_line(parser: argparse.ArgumentParser) -> tuple[str, int, list[int], float, float]:
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    return (args.p, args.nstates, args.iconds, args.excess, args.delta)


# ============<>===============
if __name__ == "__main__":
    msg = "plot_states_pops -p <path/to/output>\
     -nstates <number of states computed>\
     -iconds <List of initial conditions>\
     -excess <excess energy in eV>\
     -delta <delta energy around the excess>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True,
                        help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-nstates', type=int, required=True,
                        help='Number of states')
    parser.add_argument('-iconds', nargs='+', type=int, required=True,
                        help='List of initial conditions')
    parser.add_argument('-excess', type=float, required=True,
                        help='Excess energy in eV')
    parser.add_argument('-delta', type=float, required=True,
                        help='Delta Energy around excess')
    main(*read_cmd_line(parser))
