#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from nac.common import fs_to_nm
from nac.analysis import (autocorrelate, dephasing, spectral_density)

"""
This program plots several properties related to the interaction of a pair of states s1 and s2.
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


def plot_stuff(ens, acf, sd, deph, rate, s1, s2, ts, wsd, wdeph):
    """
    arr - a vector of y-values that are plot
    plot_mean, save_plot - bools telling to plot the mean and save the plot or not, respectively
    """
    dim_x = np.arange(ts)

    ens_av = np.average(ens, axis=2)
    ax1 = plt.subplot(321)
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Energy (eV)')
    ax1.plot(dim_x, ens_av[0, :], c='r')
    ax1.plot(dim_x, ens_av[1, :], c='b')

    ax2 = plt.subplot(322)
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Normalized AUF')
    ax2.set_ylim(-1, 1)
#    ax2.plot(ts, acf[:, 1, 0], c='r')
#    ax2.plot(ts, acf[:, 1, 1], c='b')
    ax2.plot(dim_x, acf[:, 1, 2], c='g')
    ax2.axhline(0, c="black")

    ax3 = plt.subplot(323)
    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('Un-normalized AUF')
#    ax3.plot(ts, acf[:, 1, 0], c='r')
#    ax3.plot(ts, acf[:, 1, 1], c='b')
    ax3.plot(dim_x, acf[:, 0, 2], c='g')
    ax3.axhline(0, c="black")

    ax4 = plt.subplot(324)
    ax4.set_xlabel('Time (fs)')
    ax4.set_ylabel('Dephasing (arbitrary units)')
    ax4.set_xlim(0, wdeph)
    ax4.plot(dim_x, deph[:, 0], c='r')
    ax4.plot(dim_x, deph[:, 1], c='b')

    ax5 = plt.subplot(325)
    ax5.set_xlabel('Frequency (cm-1)')
    ax5.set_ylabel('Spectral Density (arbitrary units)')
    ax5.set_xlim(0, wsd)
#    ax5.plot(sd[:, 1, 0], sd[:, 0, 0], c='r')
#    ax5.plot(sd[:, 1, 1], sd[:, 0, 1], c='b')
    ax5.plot(sd[:, 1, 2], sd[:, 0, 2], c='g')
    print('The dephasing time is : {:f} fs'.format(rate))
    print('The homogenous line broadening is  : {:f} nm'.format(1 / rate * fs_to_nm))

    fileName = "MOs.png"
    plt.savefig(fileName, format='png', dpi=300)

    plt.show()


def read_energies(path, fn, s1, s2, nconds, ts):
    inpfile = os.path.join(path, fn)
    # index of columns in the me_energy file
    cols = (s1 * 2 + 5, s2 * 2 + 5)
    xs = np.stack(np.loadtxt('{}{}'.format(inpfile, j), usecols=cols)
                  for j in range(nconds)).transpose()
    # Rows = timeframes ; Columns = states ; tensor = initial conditions
    xs = xs.swapaxes(0, 1)
    return xs


def main(path_output, s1, s2, ts, nconds, wsd, wdeph):
    ts = int(ts)
    energies = read_energies(path_output, 'me_energies', s1, s2, nconds, ts)
    # Compute the energy difference between pair of states
    d_E = energies[0:ts, 0, :] - energies[0:ts, 1, :]
    # Generate a matrix with s1, s2 and diff between them
    en_states = np.stack((energies[0:ts, 0, :], energies[0:ts, 1, :], d_E))
    # Compute autocorrelation function for each column (i.e. state)
    acf = np.stack(np.stack(autocorrelate(en_states[i, :, j])
                            for i in range(en_states.shape[0]))
                   for j in range(nconds)).transpose()
    # Average the acf over initial conditions
    acf_av = np.average(acf, axis=3)
    # Compute the spectral density for each column using the normalized acf
    sd = np.stack(spectral_density(acf_av[:, 1, i])
                  for i in range(en_states.shape[0])).transpose()
    # Compute the dephasing time for the uncorrelated acf between two states
    deph, rate = dephasing(acf_av[:, 0, 2])
    # Plot stuff
    plot_stuff(en_states, acf_av, sd, deph, rate, s1, s2, ts, wsd, wdeph)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['p', 's1', 's2', 'ts', 'nconds', 'wsd', 'wdeph']

    return [getattr(args, p) for p in attributes]

# ============<>===============
if __name__ == "__main__":

    msg = "plot_decho -p <path/to/hamiltonians> -s1 <State 1> -s2 <State 2>\
    -ts <time window for analysis> -nconds <number of initial conditions>\
    -wsd <energy window for spectral density plot in cm-1>\
    -wdeph <time window for dephasing time in fs>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True,
                        help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-s1', required=True, type=int,
                        help='Index of the first state')
    parser.add_argument('-s2', required=True, type=int,
                        help='Index of the second state')
    parser.add_argument('-ts', type=str, default='All',
                        help='Index of the second state')
    parser.add_argument('-nconds', required=True, type=int,
                        help='Number of initial conditions')
    parser.add_argument('-wsd', type=int, default=1500,
                        help='energy window for spectral density plot in cm-1')
    parser.add_argument('-wdeph', type=int, default=50,
                        help='time window for dephasing time in fs')

    main(*read_cmd_line(parser))
