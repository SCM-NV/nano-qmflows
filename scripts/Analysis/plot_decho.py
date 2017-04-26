#! /usr/bin/env python
from os.path import join 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import glob, os 
import argparse
from nac.common import (hbar, r2meV, nyq_to_cm) 
from scipy.optimize import curve_fit

def plot_stuff(ens, acf, sd, deph, s1, s2):
    """
    arr - a vector of y-values that are plot
    plot_mean, save_plot - bools telling to plot the mean and save the plot or not, respectively
    """
    dim_x = ens.shape[0]
    ts = np.arange(dim_x)

    ax1 = plt.subplot(221)
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Energy (eV)')
    ax1.plot(ts, ens[:, 0] , c='r')
    ax1.plot(ts, ens[:, 1] , c='b')

    ax2 = plt.subplot(222)
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Normalized AUF')
    ax2.plot(ts, acf[:, 1, 0], c='r')
    ax2.plot(ts, acf[:, 1, 1], c='b')
    ax2.plot(ts, acf[:, 1, 2], c='g')

    ax3 = plt.subplot(223)
    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('Dephasing (arbitrary units)')
    ax3.set_xlim(0,30)
    ax3.plot(ts, deph, c='b')

    ax4 = plt.subplot(224)
    ax4.set_xlabel('Frequency (cm-1)')
    ax4.set_ylabel('Spectral Density (arbitrary units)')
    ax4.set_xlim(0,2000)
    ax4.plot(sd[:, 1, 0], sd[:, 0, 0], c='r')
    ax4.plot(sd[:, 1, 1], sd[:, 0, 1], c='b')
    ax4.plot(sd[:, 1, 2], sd[:, 0, 2], c='g')

    fileName = "MOs.png"
    plt.savefig(fileName, format='png', dpi=300 )

    plt.show()

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def dephasing(f): 
    ts = np.arange(f.shape[0])
    cumu_ii = np.stack(np.sum(f[0:i]) for i in range(ts.size)) / hbar 
    cumu_i = np.stack(np.sum(cumu_ii[0:i]) for i in range(ts.size)) / hbar 
    deph = np.exp(-cumu_i)
    return deph 

def autocorrelate(f):
#   Compute delta_E for each state vs the mean
    d_f = f - f.mean() 
#   Compute the autocorrelation function 
    uacf = np.correlate(d_f, d_f, "full")[-d_f.size:] / d_f.size
#   Compute the normalized autocorrelation function
    nacf = uacf/uacf[0] 
    return uacf, nacf     

def spectral_density(f):
#   Fourier Transform of the nacf using a dense grid with 100000 points
    f_fft = abs(np.fft.fft(f,100000)) ** 2 
#   Fourier Transform of the time axis 
    freq = np.fft.fftfreq(len(f_fft),1)
#   Conversion of the x axis to cm-1 
    freq = freq * nyq_to_cm
    return f_fft, freq 

def read_energies(path_hams):
    files_re = glob.glob(os.path.join(path_hams, 'Ham_*_re'))
    states = np.loadtxt(files_re[0], dtype='str').shape[1] # Total number of states 
    ts = len(files_re) # Total number of time steps   
    xs = np.empty(shape=(ts, states))
    for i in range(ts):
        fn = os.path.join(path_hams,'Ham_{}_re'.format(i)) 
        xs[i] = np.diag(np.array(pd.read_csv(fn, delimiter='  ', header=None, engine='python')))
    return xs * r2meV / 1000 # return energies in eV  

def main(path_hams, s1, s2):
    energies = read_energies(path_hams) 
#   Compute the energy difference between pair of states
    d_E = energies[:, s1] - energies[:, s2]
#   Generate a matrix with s1, s2 and diff between them     
    en_states = np.column_stack((energies[:, s1], energies[:, s2], d_E))
#   Compute autocorrelation function for each column (i.e. state) 
    acf = np.stack(autocorrelate(en_states[:,i]) for i in range(en_states.shape[1])).transpose()
#   Compute the spectral density for each column using the normalized acf  
    sd = np.stack(spectral_density(acf[:, 1, i]) for i in range(en_states.shape[1])).transpose()
#   Compute the dephasing time for the uncorrelated acf between two states
    deph = dephasing(acf[:, 0, 2]) 
#   Plot stuff 
    plot_stuff(en_states, acf, sd, deph, s1, s2)

#   Get the rate of dephasing time by fitting the dephasing with a gaussian  
#    np.seterr(over='ignore')
#    popt, pcov = curve_fit(func, ts, deph)
#    xs = popt[0] * np.exp(-popt[1]*ts) + popt[2]
#    deph = np.column_stack((deph, xs)) 

#   Electron-phonon coupling strength
#       epstr = np.trapz(nacf_fft, freq) 
#       print(epstr)         

def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['p', 's1', 's2']

    return [getattr(args, p) for p in attributes]

# ============<>===============
if __name__ == "__main__":

    msg = " plot_decho -p <path/to/hamiltonians> -s1 <State 1> -s2 <State 2>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True, help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-s1', required=True, type=int, help='Index of the first state')
    parser.add_argument('-s2', required=True, type=int, help='Index of the second state')

    main(*read_cmd_line(parser))

