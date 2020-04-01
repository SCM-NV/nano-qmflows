#! /usr/bin/env python
"""
This script plots the stacked contribution of each atom type to a given MO.
You can also group together the contribution of a set of atoms
(for example that form a ligand). Usage:

plot_dos.py -ligand 2 3 -emin -5.0 -emax 2.0

When you use the ligand flag you imply that some atom kinds,
in this case k2 and k3, can be grouped together.
If you dont use the ligand flag, each atom contributes independently.
emin and emax indicates the energy window to make the plot.
"""


import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import interactive


def readatom(filename):
    # In the first line in column 6, the atom is defined
    with open(filename, 'r') as f:
        atom = f.readline().split()[6]
    return atom


def g(x_real, x_grid, delta):
    return np.exp(-2 * (x_grid - x_real) ** 2 / delta ** 2)


def convolute(x, y, x_points, sigma):
    # Compute gaussian prefactor
    prefactor = np.sqrt(2.0) / (sigma * np.sqrt(np.pi))
    # Convolute spectrum over grid
    y_points = prefactor * np.stack(
        np.sum(y * g(x, x_point, sigma)) for x_point in x_points)
    return y_points


def plot_stuff(ys, energies, legends, emin, emax):
    # In case you need more colors, this list should be expanded
    colors = ['black', 'orange', 'blue', 'red',
              'green', 'magenta', 'cyan', 'black', 'yellow']
    #   First plot
    left = 0
    for i in range(ys.shape[1]):
        plt.barh(energies, ys[:, i], height=0.02, left=left,
                 lw=0, color=colors[i], label=legends[i])
        left += ys[:, i]

    plt.figure(1)
    plt.ylim(emin, emax)
    plt.xlim(0, 1)
    plt.tick_params(direction='out')
#    plt.legend(loc='upper right').draggable()
    interactive(True)
    plt.show()

#   Prepare the cumulative density of states
    sigma = 0.2
    x_points = np.linspace(energies[0], energies[-1], 1000)
    # Take transpose to have it in (x_points, # dos) format
    dos = np.stack(convolute(
        energies, ys[:, i], x_points, sigma) for i in range(ys.shape[1])).T
    # Cumulate dos for all atoms/ligands
    cum_dos = np.stack(np.sum(dos[:, 0:i+1], axis=1)
                       for i in range(dos.shape[1])).T
    #   Plotting now
    plt.figure(2)
    ref = np.zeros(x_points.size)
    # Add a zero line for filling colors
    cum_dos = np.column_stack((ref, cum_dos))
    plt.xlim(emin, emax)
    max_y_plot = np.amax(
        cum_dos[np.where(x_points > emin) and np.where(x_points < emax), -1])
    plt.ylim(0, max_y_plot)
    for i in range(cum_dos.shape[1]-1):
        plt.fill_between(
            x_points, cum_dos[:, i], cum_dos[:, i + 1], color=colors[i], label=legends[i])
    plt.legend(loc='upper right').draggable()
    plt.tick_params(direction='out')
    interactive(False)
    plt.show()


def main(group, emin, emax):
    # Check files with PDOS
    files = sorted(glob.glob('*-k*.pdos'))
    # Define the atom type for each DOS file
    legends = [readatom(files[i]) for i in range(len(files))]
    print(files, legends)
    # MO energies
    energies = np.loadtxt(files[0], usecols=1)
    # Convert energies to eV
    energies *= 27.211
    # Occupation
    occ = np.loadtxt(files[0], usecols=2)
    lumos_indx = np.where(occ == 0)
    lumo_indx = lumos_indx[0][0]
    homo_indx = lumo_indx - 1
    hl_gap = (energies[lumo_indx] - energies[homo_indx])
    print(f'The homo-lumo gap is: {hl_gap} eV')

    # Read Files with PDOS info
    xs = [np.loadtxt(files[i]) for i in range(len(files))]
    # Add up all orbitals contribution for each atom type
    ys = np.stack(np.sum(xs[i][:, 3:], axis=1)
                  for i in range(len(files))).transpose()

    if group:
        lig_atoms = 0
        lig_name = ''
        for i in range(len(group)):
            lig_atoms += ys[:, group[i]-1]
            lig_name += legends[group[i]-1]
        ys = np.delete(ys, np.array(group)-1, axis=1)
        ys = np.column_stack((ys, lig_atoms))
        remove_atoms = [legends[i-1] for i in group]
        new_list = list(set(legends) - set(remove_atoms))
        new_list.append(lig_name)
        legends = new_list

    plot_stuff(ys, energies, legends, emin, emax)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['ligand', 'emin', 'emax']

    return [getattr(args, p) for p in attributes]


# ============<>===============
if __name__ == "__main__":

    msg = "plot_dos.py -emin <Minimum energy window> -emax <Maximum energy window>\
    -ligand <list of atoms to group>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-emin', default=-6, type=float,
                        help='Minimum energy window to plot results')
    parser.add_argument('-emax', default=-1, type=float,
                        help='Maximum energy window to plot results')
    parser.add_argument('-ligand', nargs='+', type=int,
                        help='List of atoms to group')

    main(*read_cmd_line(parser))
