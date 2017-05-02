#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import os
from nac.common import (hbar, r2meV, fs_to_cm, fs_to_nm)
from scipy.optimize import curve_fit


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
    ax3.set_ylabel('Dephasing (arbitrary units)')
    ax3.set_xlim(0, wdeph)
    ax3.plot(dim_x, deph[:, 0], c='r')
    ax3.plot(dim_x, deph[:, 1], c='b')

    ax4 = plt.subplot(325)
    ax4.set_xlabel('Frequency (cm-1)')
    ax4.set_ylabel('Spectral Density (arbitrary units)')
    ax4.set_xlim(0, wsd)
#    ax4.plot(sd[:, 1, 0], sd[:, 0, 0], c='r')
#    ax4.plot(sd[:, 1, 1], sd[:, 0, 1], c='b')
    ax4.plot(sd[:, 1, 2], sd[:, 0, 2], c='g')
    print('The dephasing rate is : {:f} fs'.format(rate))
    print('The homogenous line broadening is  : {:f} nm'.format(1 / rate * fs_to_nm))

    ax5 = plt.subplot(322)
    ax5.set_xlabel('Time (fs)')
    ax5.set_ylabel('Coupling (meV)')
    ax5.plot(dim_x, coupls[:, s1, s2], c='b')
    av_coupl = np.average(abs(coupls[:, s1, s2]))
    ax5.axhline(av_coupl, c="black")
    print('The average coupling strength is : {:f} meV'.format(av_coupl))

    fileName = "MOs.png"
    plt.savefig(fileName, format='png', dpi=300)

    plt.show()


def gauss_function(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def dephasing(f):
    ts = np.arange(f.shape[0])
    cumu_ii = np.stack(np.sum(f[0:i]) for i in range(ts.size)) / hbar
    cumu_i = np.stack(np.sum(cumu_ii[0:i]) for i in range(ts.size)) / hbar
    deph = np.exp(-cumu_i)
    np.seterr(over='ignore')
    popt, pcov = curve_fit(gauss_function, ts, deph)
    xs = popt[0] * np.exp(-(ts - popt[1]) ** 2 / (2 * popt[2] ** 2))
    deph = np.column_stack((deph, xs))
    rate = popt[2]
    return deph, rate


def autocorrelate(f):
    """
    Compute delta_E for each state vs the mean
    """
    d_f = f - f.mean()
    # Compute the autocorrelation function
    uacf = np.correlate(d_f, d_f, "full")[-d_f.size:] / d_f.size
    # Compute the normalized autocorrelation function
    nacf = uacf / uacf[0]
    return uacf, nacf


def spectral_density(f):
    """
    Fourier Transform of the nacf using a dense grid with 100000 points
    """
    f_fft = abs(1 / np.sqrt(2 * np.pi) * np.fft.fft(f, 100000)) ** 2
    # Fourier Transform of the time axis
    freq = np.fft.fftfreq(len(f_fft), 1)
    # Conversion of the x axis (given in cycles/fs) to cm-1
    freq = freq * fs_to_cm
    return f_fft, freq


def read_couplings(path_hams, ts):
    files_im = [os.path.join(path_hams, 'Ham_{}_im'.format(i)) for i in range(ts)]
    xs = np.stack(np.loadtxt(fn) for fn in files_im)
    return xs * r2meV  # return energies in meV


def read_energies(path_hams, ts):
    files_re = [os.path.join(path_hams, 'Ham_{}_re'.format(i)) for i in range(ts)]
    xs = np.stack(np.diag(np.loadtxt(fn)) for fn in files_re)
    return xs * r2meV / 1000  # return energies in eV


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
