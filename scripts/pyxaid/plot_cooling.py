#!/usr/bin/env python
"""This program plots the average electronic energy during a NAMD simulation averaged over several initial conditions.

It plots both the SH and SE population based energies.

Example:

 plot_cooling.py -p . -nstates 26 -nconds 6

Note that the number of states is the same as given in the pyxaid output.
 It must include the ground state as well.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import interactive

from nanoqm.analysis import (autocorrelate, convolute, read_energies_pyxaid,
                             read_pops_pyxaid, spectral_density)


def func(x, a, b, c, d, e):
    return a * np.exp(- x ** 2 / b ** 2) + d * np.exp(- x / e) + c

def plot_stuff(x_grid, y_grid, x_grid_scaled, y_grid_scaled, sd, w_en, w_en_scaled, nconds, outs, energies, ts, dt):

    plt.figure(1)
    plt.xlabel('Time (fs)')
    plt.ylabel('Energy (eV)')
    for iconds in range(nconds):
        plt.imshow(y_grid[iconds, :, :].T, aspect='auto', extent=(0, len(ts)*dt, np.min(x_grid), np.max(x_grid)), origin='lower', interpolation='bicubic', cmap='hot')
    plt.plot(ts * dt, w_en, 'w')
    interactive(True)
    plt.show()

    plt.figure(2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Excess Energy (eV)')
    for iconds in range(nconds):
        plt.imshow(y_grid_scaled[iconds, :, :].T, aspect='auto', extent=(0, len(ts)*dt, np.min(x_grid_scaled), np.max(x_grid_scaled)), origin='lower', interpolation='bicubic', cmap='hot')  
    plt.plot(ts * dt, w_en_scaled, 'w')
    interactive(True)
    plt.show()

    plt.figure(3) 
    plt.xlabel('Time (fs)')
    plt.ylabel('State Number')
    for iconds in range(nconds):
        plt.imshow(outs[:, :, iconds].T, aspect='auto', origin='lower', extent = ( 0, len(ts) * dt, 0, outs.shape[1]), interpolation='bicubic', cmap='hot')    
    interactive(True)
    plt.show() 

    plt.figure(4)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Counts')
    plt.hist(w_en, bins='auto')
    interactive(True)
    plt.show()

    plt.figure(5)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Counts')
    plt.hist(w_en_scaled, bins='auto')
    interactive(True)
    plt.show()

    plt.figure(6)
    plt.xlabel('Time (fs)')
    plt.ylabel('State Energy (eV)')
    plt.plot(ts*dt, energies[:, :, 0])
    interactive(True)
    plt.show()

    plt.figure(7)
    plt.ylabel('State Number')
    plt.xlabel('Freqencies cm-1')
    sd_int = sd[:, 0, :int(sd.shape[2]/2)] 
    sd_freq = sd[0, 1, :int(sd.shape[2]/2)]
    plt.imshow(sd_int, aspect='auto', origin='lower', extent = (np.min(sd_freq), np.max(sd_freq), 0, sd_int.shape[0] ), interpolation='bicubic', cmap='hot')    
    interactive(False)
    plt.show()


def main(path_output, nstates, nconds, dt, sigma):
    outs = read_pops_pyxaid(path_output, 'out', nstates, nconds)
    energies = read_energies_pyxaid(path_output, 'me_energies', nstates, nconds)
    ##################################
    # Remove the ground state and scale the energies to the lowest excitation energy
    outs = outs[:, 1:, :]
    energies = energies[:, 1:, :]
    lowest_hl_gap = np.amin(energies, axis=1)
    highest_hl_gap = np.amax(energies, axis=1)
    # Convolute a gaussian for each timestep and for each initial condition
    x_grid = np.linspace(np.min(lowest_hl_gap), np.max(highest_hl_gap), 100)
    y_grid = np.stack(np.stack(
        convolute(energies[time, :, iconds], outs[time, :, iconds], x_grid, sigma)
        for time in range(energies.shape[0]) )
        for iconds in range(nconds) )   
    # Scale the energy for computing the excess energy plots
    energies_scaled = np.stack(energies[:, istate, :] - lowest_hl_gap for istate in range(nstates-1)) 
    energies_scaled = energies_scaled.swapaxes(0,1) # Just reshape to have the energies shape (time, nstates, nconds) 
    x_grid_scaled = np.linspace(0, np.max(energies_scaled), 100)
    y_grid_scaled = np.stack(np.stack(
        convolute(energies_scaled[time, :, iconds], outs[time, :, iconds], x_grid_scaled, sigma)
        for time in range(energies_scaled.shape[0]) )
        for iconds in range(nconds) ) 
    # This part is done
    ##################################
    # Compute weighted energies vs population at a given time t
    eav_outs = energies * outs
    el_ene_outs = np.sum(eav_outs, axis=1)
    # Scale them to the lowest excitation energy
    ene_outs_ref0 = el_ene_outs - lowest_hl_gap
    #Average over initial conditions 
    ene_outs_ref0 = np.average(ene_outs_ref0, axis=1) 
    el_ene_av = np.average(el_ene_outs, axis = 1) 
    ##################################
    # Compute autocorrelation function for consecutive pair of states 
    d_en = np.stack(energies[:, istate, 0] - energies[:, istate-1, 0] for istate in range(1, nstates-1))
    acf = np.stack(autocorrelate(d_en[istate, :]) for istate in range(d_en.shape[0]))
    # Compute spectral density
    nacf = acf[:, 1, :]
    sd = np.stack(spectral_density(nacf[istate, :], dt) for istate in range(d_en.shape[0]))
    #################################
    # Call plotting function
    ts = np.arange(energies.shape[0])
    plot_stuff(x_grid, y_grid, x_grid_scaled, y_grid_scaled, sd, el_ene_av, ene_outs_ref0, nconds, outs, energies, ts, dt)

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
      -dt <nuclear time step>\
      -sigma <line broadening for convoluton>"
      
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True,
                        help='path to the Hamiltonian files in Pyxaid format')
    parser.add_argument('-nstates', type=int, required=True,
                        help='Number of states')
    parser.add_argument('-nconds', type=int, required=True,
                        help='Number of initial conditions')
    parser.add_argument('-dt', type=float, required=False, default=1.0,
                        help='Nuclear Time Step')
    parser.add_argument('-sigma', type=float, required=False, default=0.05,
                        help='Line broadening for the convolution with Gaussian functions')
    main(*read_cmd_line(parser))
