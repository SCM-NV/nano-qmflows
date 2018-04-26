#!/usr/bin/env python
"""
This program plots the average electronic energy during a NAMD simulatons
averaged over several initial conditions.
It plots both the SH and SE population based energies.

Example:

 plot_cooling.py -p . -nstates 26 -nconds 6

Note that the number of states is the same as given in the pyxaid output.
 It must include the ground state as well.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import interactive
import argparse
from scipy.optimize import curve_fit
from nac.analysis import (read_energies_pyxaid, read_pops_pyxaid, convolute, autocorrelate, dephasing, spectral_density)

def func(x, a, b, c, d, e):
    return a * np.exp(- x ** 2 / b ** 2 ) + d * np.exp(- x / e ) + c

def plot_stuff(ts, en, en_scaled, w_en, w_en_scaled, aven, minim, maxim, start, end, uacf, nacf, sd_int, sd_freq, dt):
  
    plt.figure(1)   
    plt.xlabel('Time (fs)')
    plt.ylabel('Energy (eV)')
    for i in range(en.shape[0]): 
        plt.imshow(en[i, :, :], aspect='auto', origin='lower', extent=(0, len(ts) * dt, minim, maxim), interpolation='bicubic', cmap='hot')
    plt.plot(ts, w_en, 'w')
    interactive(True) 
    plt.show() 
    
    plt.figure(2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Scaled Energy (eV)')
    for i in range(en_scaled.shape[0]):
        plt.imshow(en_scaled[i, :, :], aspect='auto', origin='lower', extent=(0, len(ts) * dt, 0, np.max(en_scaled)), interpolation='bicubic', cmap='hot')
    plt.plot(ts, w_en_scaled, 'w')
    interactive(True)
    plt.show() 

    plt.figure(3)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Counts') 
    plt.hist(w_en, bins='auto') 
    interactive(True)
    plt.show() 

    plt.figure(4)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Counts')
    plt.hist(w_en_scaled, bins='auto') 
    interactive(True)
    plt.show()
   
    plt.figure(5)
    plt.xlabel('Time')
    plt.ylabel('Energy (eV)')
    plt.plot(ts, aven) 
    interactive(True)
    plt.show()

    plt.figure(6)
    ts = np.arange(uacf.size) * dt 
    plt.xlabel('Un-normalized AUC')
    plt.ylabel('Time')
    plt.plot(ts, uacf) 
    interactive(True)
    plt.show()

    plt.figure(7)
    plt.xlabel('Normalized AUC')
    plt.ylabel('Time')
    plt.plot(ts, nacf)           
    interactive(True)
    plt.show()

    plt.figure(8)
    plt.xlabel('Spectral Density')
    plt.ylabel('Freqencies cm-1')
    plt.xlim(0, 300) 
    plt.ylim(0, np.max(sd_int))  
    plt.plot(sd_freq, sd_int)
    interactive(False)
    plt.show()
    
def main(path_output, nstates, nconds, dt, sigma):
    outs = read_pops_pyxaid(path_output, 'out', nstates, nconds)
    energies = read_energies_pyxaid(path_output, 'me_energies', nstates, nconds)
    lowest_hl_gap = np.amin(energies[:, 1:, :], axis=1)
    highest_hl_gap = np.amax(energies[:, 1:, :], axis=1)
    energies_scaled = np.stack((energies[:, i, :] - lowest_hl_gap) for i in range(1, nstates))
    energies_scaled = energies_scaled.swapaxes(0, 1) 
    outs = outs[:, 1:, :]
    energies = energies[:, 1:, :]
    # Convolute a gaussian for each timestep
    x_grid = np.linspace(np.min(lowest_hl_gap),np.max(highest_hl_gap), 100)  
    y_grid = np.stack(np.stack(np.stack(convolute(energies[itime, :, iconds], outs[itime, :, iconds], x_grid, sigma) for itime in range(energies.shape[0]) ) for iconds in range(nconds)))
    x_grid_scaled = np.linspace(0, np.max(energies_scaled), 100)  
    y_grid_scaled = np.stack(np.stack(np.stack(convolute(energies_scaled[itime, :, iconds], outs[itime, :, iconds], x_grid_scaled, sigma) for itime in range(energies_scaled.shape[0]) ) for iconds in range(nconds))) 
    # This part is done
    ##################################
    # Compute weighted energies vs population at a given time t
    eav_outs = energies * outs
    # Ensamble average over initial conditions of the electronic energy
    # as a function of time
    el_ene_outs = np.average(np.sum(eav_outs, axis=1), axis=1)
    # Scale them to the lowest excitation energy
#    ene_outs_ref0 = el_ene_outs - lowest_hl_gap
    # Compute autocorrelation function of the weighted energy
    uacf, nacf = autocorrelate(el_ene_outs)
    # Compute spectral density
    sd_int, sd_freq = spectral_density(nacf, dt)
    #################################
    # Print the 3d-array on a file for plotting 
#    xs = ""
#    for time in range(av_energies_scaled.shape[0]):
#        for x_point in range(x_grid.size):
#             xs += '{:f} {:f} {:f} \n'.format(time, x_grid[x_point], y_grid[time, x_point])
#    with open('cooling_plot_data.txt', 'w') as f:
#        f.write(xs)  
    # Plotting stuff
    # Define size axes for all plot 
    ts = np.arange(energies_scaled.shape[0]) * dt 
    # Call plotting function 
    plot_stuff(ts, y_grid, y_grid_scaled, el_ene_outs, ene_outs_ref0, av_energies, np.min(lowest_hl_gap), np.max(highest_hl_gap), uacf, nacf, sd_int, sd_freq, dt)

def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['p', 'nstates', 'nconds', 'dt', 'sigma']

    return [getattr(args, p) for p in attributes]


# ============<>===============
if __name__ == "__main__":
    msg = "plot_states_pops -p <path/to/output>\
     -nstates <number of states computed>\
      -nconds <number of initial conditions>\
      -dt <nuclear time step>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True,
                        help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-nstates', type=int, required=True,
                        help='Number of states')
    parser.add_argument('-nconds', type=int, required=True,
                        help='Number of initial conditions')
    parser.add_argument('-dt', type=float, required=False, default = 1.0, 
                        help='Nuclear Time Step')
    parser.add_argument('-sigma', type=float, required=False, default=0.05, 
                        help='Line broadening for the convolution with Gaussian functions')
    main(*read_cmd_line(parser))
