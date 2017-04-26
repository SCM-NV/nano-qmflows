#! /usr/bin/env python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit

from interactive import ask_question


def plot_stuff(ens, uacf, nacf, nacf_fft, freq, deph, states):
    """
    arr - a vector of y-values that are plot
    plot_mean, save_plot - bools telling to plot the mean and save the plot or not, respectively
    """
    dim_x = ens.shape[0]
    ts = np.arange(dim_x)

    ax1 = plt.subplot(221)
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Energy (eV)')
    ax1.plot(ts, ens[:, 0] , c='r', label='Energy_{:d}'.format(states[0]))
    ax1.plot(ts, ens[:, 1] , c='b', label='Energy_{:d}'.format(states[1]))

    ax2 = plt.subplot(222, sharex=ax1)
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Normalized AUF')
    ax2.plot(ts, nacf[:, 0], c='r')
    ax2.plot(ts, nacf[:, 1], c='b')
    ax2.plot(ts, nacf[:, 2], c='g')

    ax3 = plt.subplot(223)
    ax3.set_xlabel('Frequency (cm-1)')
    ax3.set_ylabel('Spectral Density (arbitrary units)')
    ax3.set_xlim(0,2000)
    ax3.plot(freq, nacf_fft[:, 0], c='r')
    ax3.plot(freq, nacf_fft[:, 1], c='b')
    ax3.plot(freq, nacf_fft[:, 2], c='g')

    ax4 = plt.subplot(224)
    ax4.set_xlabel('Time (fs)')
    ax4.set_ylabel('Dephasing (arbitrary units)')
    ax4.set_xlim(0,30)
    ax4.plot(ts, deph[:,0], c='b')
    ax4.plot(ts, deph[:,1], c='r')

    fileName = "MOs.png"
    plt.savefig(fileName, format='png', dpi=300 )

    plt.show()

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def read_energies(fn, states):
    xs = np.diag(np.array(pd.read_csv(fn, delimiter='  ', header=None, engine='python')))
    return xs[states] 

def main():
    r2meV = 13605.698  # conversion from rydberg to meV
    nyq_to_cm = 2 * np.pi * 100000 / 3 
    hbar = 6.582119e-16 # eV * s 
    hbar = 0.6582119 # eV * fs 
    print("The labeling is as\n...\n98 HOMO-1\n99 HOMO\n100 LUMO\n101 LUMO+1\n...")
    m1 = ask_question("Please define the first MO (integer). [Default: 101] ",
                      special='int', default='101')
    m2 = ask_question("Please define the second MO (integer). [Default: 100] ",
                      special='int', default='100')

    files_re = glob.glob('Ham_*_re')
#    files_im = glob.glob('Ham_*_im')
    pair = [m1, m2] 
    if files_re:
        energies = np.stack(read_energies('Ham_{}_re'.format(i), pair) for i in range(len(files_re))) * r2meV / 1000 
        delta_E = energies[:,0] - energies[:,1]
        energies = np.column_stack((energies, delta_E))
#   Compute the energy differences 
        d_delta_E = np.stack(energies[:,i] - energies[:,i].mean() for i in range(len(pair) + 1)).transpose() 
#   Unnormalized autocorrelation function between two states
        uacf = np.stack(np.correlate(d_delta_E[:,i], d_delta_E[:,i], "full") for i in range(len(pair) + 1)).transpose()
        uacf = uacf[int(uacf.shape[0]/2):] / d_delta_E.shape[0]
#   Normalized autocorrelation function 
        nacf = np.stack(uacf[:,i]/uacf[0,i] for i in range(len(pair) + 1)).transpose()
#   Spectral Density 
        nacf_fft = np.stack(abs(np.fft.fft(nacf[:,i],100000)) ** 2 for i in range(len(pair) + 1)).transpose()
        freq = np.fft.fftfreq(len(nacf_fft),1)
        freq = freq * nyq_to_cm
#   Dephasing between the two states 
        ts = np.arange(uacf.shape[0]) 
        cumu_ii = np.stack(np.sum(uacf[0:i,2]) for i in range(ts.size)) / hbar 
        cumu_i = np.stack(np.sum(cumu_ii[0:i]) for i in range(ts.size)) / hbar
        deph = np.exp(-cumu_i) 
#   Get the rate of dephasing time 
        np.seterr(over='ignore')
        popt, pcov = curve_fit(func, ts, deph)
        xs = popt[0] * np.exp(-popt[1]*ts) + popt[2]
        deph = np.column_stack((deph, xs)) 

#   Electron-phonon coupling strength
#       epstr = np.trapz(nacf_fft, freq) 
#       print(epstr)         
        plot_stuff(energies, uacf, nacf, nacf_fft, freq, deph, pair)
    else:
        print('ERROR: No files found. Please make sure that you are in the '
              'hamiltonians directory.')
# ============<>===============
if __name__ == "__main__":
    main()
