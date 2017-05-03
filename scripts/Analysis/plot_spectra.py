#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import os
from nac.common import (hbar, r2meV, fs_to_cm, fs_to_nm)
from nac.analysis import (autocorrelate, dephasing, gauss_function, read_couplings, read_energies, spectral_density) 


def plot_stuff(ens, coupls, acf, sd, deph, rate, s1, s2, ts, wsd, wdeph):
    """
    arr - a vector of y-values that are plot
    plot_mean, save_plot - bools telling to plot the mean and save the plot or not, respectively
    """
    dim_x = np.arange(ts)

    ax1 = plt.subplot(321)
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Energy (eV)')
    ax1.plot(dim_x, ens[:, 0], c='r')
    ax1.plot(dim_x, ens[:, 1], c='b')

    ax2 = plt.subplot(323)
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Normalized AUF')
#    ax2.plot(ts, acf[:, 1, 0], c='r')
#    ax2.plot(ts, acf[:, 1, 1], c='b')
    ax2.plot(dim_x, acf[:, 1, 2], c='g')
    ax2.axhline(0, c="black")

    ax3 = plt.subplot(324)
    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('Un-normalized AUF')
#    ax3.plot(ts, acf[:, 1, 0], c='r')
#    ax3.plot(ts, acf[:, 1, 1], c='b')
    ax3.plot(dim_x, acf[:, 0, 2], c='g')
    ax3.axhline(0, c="black")

    ax4 = plt.subplot(326)
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

    ax6 = plt.subplot(322)
    ax6.set_xlabel('Time (fs)')
    ax6.set_ylabel('Coupling (meV)')
    ax6.plot(dim_x, coupls[:, s1, s2], c='b')
    av_coupl = np.average(abs(coupls[:, s1, s2]))
    ax6.axhline(av_coupl, c="black")
    print('The average coupling strength is : {:f} meV'.format(av_coupl))

    fileName = "MOs.png"
    plt.savefig(fileName, format='png', dpi=300)

    plt.show()

def main(path_hams, s1, s2, ts, wsd, wdeph):
    if ts == 'All':
        files = glob.glob(os.path.join(path_hams, 'Ham_*_re'))
        ts = len(files)
    else:
        ts = int(ts)
    energies = read_energies(path_hams, ts)
    couplings = read_couplings(path_hams, ts)
    # Compute the energy difference between pair of states
    d_E = energies[:, s1] - energies[:, s2]
    # Generate a matrix with s1, s2 and diff between them
    en_states = np.column_stack((energies[:, s1], energies[:, s2], d_E))
    # Compute autocorrelation function for each column (i.e. state)
    acf = np.stack(autocorrelate(en_states[:, i]) for i in range(en_states.shape[1])).transpose()
    # Compute the spectral density for each column using the normalized acf
    sd = np.stack(spectral_density(acf[:, 1, i]) for i in range(en_states.shape[1])).transpose()
    # Compute the dephasing time for the uncorrelated acf between two states
    deph, rate = dephasing(acf[:, 0, 2])
    # Plot stuff
    plot_stuff(en_states, couplings, acf, sd, deph, rate, s1, s2, ts, wsd, wdeph)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['p', 's1', 's2', 'ts', 'wsd', 'wdeph']

    return [getattr(args, p) for p in attributes]

# ============<>===============
if __name__ == "__main__":

    msg = " plot_decho -p <path/to/hamiltonians> -s1 <State 1> -s2 <State 2> -ts <time window for analysis> -wsd <energy window for spectral density plot in cm-1> -wdeph <time window for dephasing time in fs>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True, help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-s1', required=True, type=int, help='Index of the first state')
    parser.add_argument('-s2', required=True, type=int, help='Index of the second state')
    parser.add_argument('-ts', type=str, default='All', help='Index of the second state')
    parser.add_argument('-wsd', type=int, default=1500,
                        help='energy window for spectral density plot in cm-1')
    parser.add_argument('-wdeph', type=int, default=50, help='time window for dephasing time in fs')

    main(*read_cmd_line(parser))
