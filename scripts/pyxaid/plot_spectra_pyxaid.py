#! /usr/bin/env python
"""This program plots several properties related to the interaction of a pair of states s1 and s2.

1. The energies of the two states along the MD trajectory
2. The non-adiabatic coupling between the two states (in meV)
3. The normalized autocorrelation function (NAUF) of a pair of entangled states
 during the MD trajectory.
   It describes how the interaction between two states depend on earlier time.
   The AUF of poorly correlated states goes rapidly to zero.
4. The unnormalized autocorrelation function. It provides similar infos than NAUF,
  but the starting value is average oscillation at t=0 (in eV**2)
5. The spectral density computed from the fourier transform of the NAUF.
 It identifies which phonones coupled to the electronic subsystem.
 In this case, it tells which phonons are involved in the coupling between the
 pair of entangles states.
6. The dephasing function. It indicates how the two entangled states
 decohere over time.
 The inverse of the decoherence rate is related to the homogenoues line
 broadening of emission lines when computed between HOMO and LUMO states.

Note that you have to provide the location of the folder where the NAMD
 hamiltonian elements are stored using the -p flag.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import interactive

from nanoqm import logger
from nanoqm.analysis import autocorrelate, dephasing, spectral_density
from nanoqm.common import fs_to_nm


def plot_stuff(ens, acf, sd, deph, rate, s1, s2, dt, wsd, wdeph):
    """
    arr - a vector of y-values that are plot
    plot_mean, save_plot - bools telling to plot the mean and save the plot or not, respectively
    """
    dim_x = np.arange(ens.shape[1]) * dt

    plt.figure(1)
    plt.title(f'Energies of state {s1} (red) and {s2} (blue)')
    plt.xlabel('Time (fs)')
    plt.ylabel('Energy (eV)')
    plt.plot(dim_x, ens[0, :], c='r')
    plt.plot(dim_x, ens[1, :], c='b')
    interactive(True)
    plt.show()

    plt.figure(2)
    plt.title(f'Normalized AUF between state {s1} and {s2}')
    plt.xlabel('Time (fs)')
    plt.ylabel('Normalized AUF')
    plt.ylim(-1, 1)
#    plt.plot(ts, acf[:, 1, 0], c='r')
#    plt.plot(ts, acf[:, 1, 1], c='b')
    plt.plot(dim_x, acf[:, 1, 2], c='g')
    plt.axhline(0, c="black")
    interactive(True)
    plt.show()

    plt.figure(3)
    plt.title(f'Un-normalized AUF between state {s1} and {s2}')
    plt.xlabel('Time (fs)')
    plt.ylabel('Un-normalized AUF')
#    plt.plot(ts, acf[:, 1, 0], c='r')
#    plt.plot(ts, acf[:, 1, 1], c='b')
    plt.plot(dim_x, acf[:, 0, 2], c='g')
    plt.axhline(0, c="black")
    interactive(True)
    plt.show()

    plt.figure(4)
    plt.title(f'Dephasing time between state {s1} and {s2}')
    plt.xlabel('Time (fs)')
    plt.ylabel('Dephasing (arbitrary units)')
    plt.xlim(0, wdeph)
    plt.plot(dim_x, deph[:, 0], c='r')
    plt.plot(dim_x, deph[:, 1], c='b')
    logger.info(f'The dephasing time is : {rate:f} fs')
    line_broadening = 1 / rate * fs_to_nm
    logger.info(f'The homogenous line broadening is  : {line_broadening:f} nm')
    interactive(True)
    plt.show()

    plt.figure(5)
    plt.title(f'Influence spectrum state {s1}')
    plt.xlabel('Frequency (cm-1)')
    plt.ylabel('Spectral Density (arbitrary units)')
    plt.xlim(0, wsd)
    plt.plot(sd[0, 1, :], sd[0, 0, :], c='g')
    interactive(True)
    plt.show()

    plt.figure(6)
    plt.title(f'Influence spectrum state {s2}')
    plt.xlabel('Frequency (cm-1)')
    plt.ylabel('Spectral Density (arbitrary units)')
    plt.xlim(0, wsd)
    plt.plot(sd[1, 1, :], sd[1, 0, :], c='g')
    interactive(True)
    plt.show()

    plt.figure(7)
    plt.title(f'Influence spectrum across state {s1} and {s2}')
    plt.xlabel('Frequency (cm-1)')
    plt.ylabel('Spectral Density (arbitrary units)')
    plt.xlim(0, wsd)
    plt.plot(sd[2, 1, :], sd[2, 0, :], c='g')
    interactive(False)
    plt.show()
    fileName = "MOs.png"
    plt.savefig(fileName, format='png', dpi=300)


def main(path_output, s1, s2, dt, wsd, wdeph):
    fn = 'me_energies0'  # it is only necessary the first initial condition
    inpfile = os.path.join(path_output, fn)
    cols = (s1 * 2 + 5, s2 * 2 + 5)
    energies = np.loadtxt(inpfile, usecols=cols)
    # Compute the energy difference between pair of states
    d_E = energies[:, 0] - energies[:, 1]
    # Generate a matrix with s1, s2 and diff between them
    en_states = np.stack((energies[:, 0], energies[:, 1], d_E))
    # Compute autocorrelation function for each column (i.e. state)
    # Take the transpose to have the correct shape for spectral_density
    acf = np.stack(autocorrelate(en_states[i, :])
                   for i in range(en_states.shape[0])).T
    # Compute the spectral density for each column using the normalized acf
    sd = np.stack(spectral_density(acf[:, 1, i], dt)
                  for i in range(en_states.shape[0]))
    # Compute the dephasing time for the uncorrelated acf between two states
    deph, rate = dephasing(acf[:, 0, 2], dt)
    # Plot stuff
    plot_stuff(en_states, acf, sd, deph, rate, s1, s2, dt, wsd, wdeph)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['p', 's1', 's2', 'dt', 'wsd', 'wdeph']

    return [getattr(args, p) for p in attributes]


# ============<>===============
if __name__ == "__main__":

    msg = "plot_decho -p <path/to/hamiltonians> -s1 <State 1> -s2 <State 2>\
    -dt <time step in fs> \
    -wsd <energy window for spectral density plot in cm-1>\
    -wdeph <time window for dephasing time in fs>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True,
                        help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-s1', required=True, type=int,
                        help='Index of the first state')
    parser.add_argument('-s2', required=True, type=int,
                        help='Index of the second state')
    parser.add_argument('-dt', type=float, default=1.0,
                        help='Index of the second state')
    parser.add_argument('-wsd', type=int, default=1500,
                        help='energy window for spectral density plot in cm-1')
    parser.add_argument('-wdeph', type=int, default=50,
                        help='time window for dephasing time in fs')

    main(*read_cmd_line(parser))
