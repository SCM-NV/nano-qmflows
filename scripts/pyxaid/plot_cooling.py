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
import matplotlib.pyplot as plt
from matplotlib import interactive
import argparse
from nac.analysis import (read_energies_pyxaid, read_pops_pyxaid, convolute, autocorrelate, spectral_density)


def func(x, a, b, c, d, e):
    return a * np.exp(- x ** 2 / b ** 2) + d * np.exp(- x / e) + c

#def read_energies(path, fn, nstates, nconds):
#    inpfile = os.path.join(path, fn)
#    cols = tuple(range(5, nstates * 2 + 5, 2))
#    xs = np.stack(np.loadtxt('{}{}'.format(inpfile, j), usecols=cols)
#                  for j in range(nconds)).transpose()
#    # Rows = timeframes ; Columns = states ; tensor = initial conditions
#    xs = xs.swapaxes(0, 1)
#    return xs


#def read_pops(path, fn, nstates, nconds):
#    inpfile = os.path.join(path, fn)
#    cols = tuple(range(3, nstates * 2 + 3, 2))
#    xs = np.stack(np.loadtxt('{}{}'.format(inpfile, j), usecols=cols)
#                  for j in range(nconds)).transpose()
#    # Rows = timeframes ; Columns = states ; tensor = initial conditions
#    xs = xs.swapaxes(0, 1)
#    return xs

#def g(x_real, x_grid, delta):
#    return np.exp(-2 * (x_grid - x_real) ** 2 / delta ** 2)

#def convolute(x, y, x_points, sigma):
#    # Compute gaussian prefactor
#    prefactor = np.sqrt(2.0) / (sigma * np.sqrt(np.pi)) 
#    # Convolute spectrum over grid 
#    y_points = prefactor * np.stack(np.sum( y * g(x, x_point, sigma) ) for x_point in x_points)
#    return y_points 

def plot_stuff(ts, aven_conv, aven_scaled_conv, w_en, w_en_scaled, aven, minim, maxim, start, end, uacf, nacf, sd_int, sd_freq, dt):

    plt.figure(1)
    plt.xlabel('Time (fs)')
    plt.ylabel('Energy (eV)')
    plt.imshow(aven_conv.T, aspect='auto', origin='lower', extent=(0, len(ts) * dt, minim, maxim), interpolation='bicubic', cmap='hot')
    plt.plot(ts, w_en, 'w')
    interactive(True)
    plt.show()

    plt.figure(2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Scaled Energy (eV)')
    plt.imshow(aven_scaled_conv.T, aspect='auto', origin='lower', extent=(0, len(ts) * dt, start, end), interpolation='bicubic', cmap='hot')
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


def main(path_output, nstates, nconds, dt, start, end, sigma):
    outs = read_pops_pyxaid(path_output, 'out', nstates, nconds)
    energies = read_energies_pyxaid(path_output, 'me_energies', nstates, nconds)
    ##################################
    # Averaged energies and populations over initial conditions
    av_outs = np.average(outs, axis=2)
    av_energies = np.average(energies, axis=2)
    # Remove the ground state and scale the energies to the lowest excitation energy
    av_outs = av_outs[:, 1:]
    av_energies = av_energies[:, 1:]
    lowest_hl_gap = np.average(np.amin(energies[:, 1:, :], axis=1), axis=1)
    highest_hl_gap = np.average(np.amax(energies[:, 1:, :], axis=1), axis=1)
    av_energies_scaled = (av_energies.transpose() - lowest_hl_gap).transpose()
    # Convolute a gaussian for each timestep
    x_grid = np.linspace(np.min(lowest_hl_gap), np.max(highest_hl_gap), 100)
    y_grid = np.stack(np.stack(
        convolute(av_energies[time, :], av_outs[time, :], x_grid, sigma)
        for time in range(av_energies.shape[0])))
    x_grid_scaled = np.linspace(start, end, 100)
    y_grid_scaled = np.stack(np.stack(
        convolute(av_energies_scaled[time, :], av_outs[time, :], x_grid_scaled, sigma)
        for time in range(av_energies_scaled.shape[0])))
    # This part is done
    ##################################
    # Compute weighted energies vs population at a given time t
    eav_outs = energies * outs
    # Ensamble average over initial conditions of the electronic energy
    # as a function of time
    el_ene_outs = np.average(np.sum(eav_outs, axis=1), axis=1)
    # Scale them to the lowest excitation energy
    ene_outs_ref0 = el_ene_outs - lowest_hl_gap
    # Compute autocorrelation function of the weighted energy
    uacf, nacf = autocorrelate(ene_outs_ref0)
    # Compute spectral density
    sd_int, sd_freq = spectral_density(nacf, dt)
    #################################
    # Print the 3d-array on a file for plotting
    xs = ""
    for time in range(av_energies_scaled.shape[0]):
        for x_point in range(x_grid.size):
            xs += '{:f} {:f} {:f} \n'.format(time, x_grid[x_point], y_grid[time, x_point])
    with open('cooling_plot_data.txt', 'w') as f:
        f.write(xs)
    # Plotting stuff
    # Define size axes for all plot
    ts = np.arange(av_energies.shape[0]) * dt
    # Call plotting function
    plot_stuff(ts, y_grid, y_grid_scaled, el_ene_outs, ene_outs_ref0, av_energies, np.min(lowest_hl_gap), np.max(highest_hl_gap), start, end, uacf, nacf, sd_int, sd_freq, dt)

#    plt.plot(ts, ene_outs_ref0, 'w')
#    plt.show()
#    plt.hist(el_ene_outs[1000:], bins='auto')
#    plt.show()
#    plt.imshow(av_outs.T, aspect='auto', origin='lower', extent=(0, av_energies.shape[0] * dt, 0, av_energies.shape[1] * dt), interpolation='bicubic', cmap='hot')
#    plt.show() 


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['p', 'nstates', 'nconds', 'dt', 'start', 'end', 'sigma']

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
    parser.add_argument('-dt', type=float, required=False, default=1.0,
                        help='Nuclear Time Step')
    parser.add_argument('-start', type=float, required=False, default=0.0,
                        help='Start point for the convolution on the energy axis')
    parser.add_argument('-end', type=float, required=False, default=3.0,
                        help='End point for the convolution on the energy axis')
    parser.add_argument('-sigma', type=float, required=False, default=0.05,
                        help='Line broadening for the convolution with Gaussian functions')
    main(*read_cmd_line(parser))
