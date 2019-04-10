#! /usr/bin/env python
"""
This script convolutes the calculated oscillator strengts of multiple structures and plot the average spectrum for this structures.
Usage:
convolution.py -sigma 0.05 -n 2
Use the sigma flag to change the sigma parameter of the gaussian functions used in the convolution.
If you use the n flag you imply that you want to plot only the spectrum of structure number n (starting from 0)instead of an average over all strucutures.
"""

import numpy as np
from nac.analysis import convolute, func_conv
import matplotlib.pyplot as plt
import glob
import argparse

def main(sigma, n):
    # Check output files 
    files = sorted(glob.glob('output_*.txt'))
    if n is None:
       # Define energy window for the plot
       energy = np.loadtxt(files[0], usecols=1)
       emax = energy [-1] + 0.5
       emin = energy [0] - 0.5
       x_grid = np.linspace(emin, emax, 800)
       y_grid = np.empty([x_grid.size, 0])
       for f in files:
           # Read transition energies and oscillator strengths
           data = np.loadtxt(f, usecols=(1,2))
           # Convolute each spectrum at a time
           ys = convolute(data[:, 0], data[:, 1], x_grid, sigma)
           # Stack the spectra in a matrix 
           y_grid = np.column_stack((y_grid, ys))
       # Average them and plot 
       y_grid = np.sum(y_grid, axis=1) / len(files)
       plt.plot(x_grid, y_grid)
    else:
       # Read transition energies and oscillator strengths
       data = np.loadtxt(files[n], usecols=(1, 2))
       # Define energy window for the plot
       emax = data [-1, 0] + 0.5
       emin = data [0, 0] - 0.5
       x_grid = np.linspace(emin, emax, 800)
       # Convolute and plot
       y_grid = convolute(data[:, 0], data[:, 1], x_grid, sigma)
       plt.plot(x_grid, y_grid)

    plt.xlabel('Energy[eV]')
    plt.ylabel('Oscillator strength')
    plt.show()


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['sigma', 'n']

    return [getattr(args, p) for p in attributes]


# ============<>===============
if __name__ == "__main__":

    msg = "convolution.py -sigma <sigma parameter of the gaussian functions> -n <plot only the spectrum of structure n>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-sigma', default=0.1, type=float,
                        help='Sigma parameter of the gaussian functions')
    parser.add_argument('-n', default=None, type=int,
                        help='Plot only the spectrum of the strucure number n')

    main(*read_cmd_line(parser))

