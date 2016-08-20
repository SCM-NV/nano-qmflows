#! /usr/bin/env python

from matplotlib.backends.backend_pdf import PdfPages

import datetime
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import os
from interactive import ask_question


msg = ('This is a program that plots the spectral density for a certain'
       'pair of states. Usage: Make sure that you are in the out folder containing the'
       'icond-files and fill in the prompted questions.')


def ask_for_states():
    print("""The states are labeled as in PYXAID; the first state being 0.
    You can look them up in the output.""")
    i1 = ask_question("What is the integer representing the first state (int)? ",
                      special='int')
    i2 = ask_question("What is the integer representing the second state (int)? ",
                      special='int')
    return i1, i2


def read_files(i1, i2):
    """
    function that opens all the spectral density files of all the initial
    conditions for states i1 and i2
    and returns
    w - which is a list containing the w-values, which should be the same
    for all initial conditions
    w - type: list of floats
    J - containing lists of the J values for all initial conditions
    J - type: list of lists of floats
    """
    w, j = [], []
    files = os.listdir('.')
    name = 'icond*pair{:d}_{:d}Spectral_density.txt'.format(i1, i2)
    density_files = fnmatch.filter(files, name)
    if density_files:
        for filename in density_files:
            arr = np.loadtxt(filename, usecols=(3, 5))
            arr = np.transpose(arr)
            w.append(arr[0])
            j.append(arr[1])
        return w, j
    else:
        name2 = name[0:4] + '0' + name[6:]
        msg = ('File not found.\nAre you in the out folder? And are '
               'you sure the ints are correct?\n'
               'The program was looking for a file named: \'{}\' in your '
               'current directory.'.format(name2))
        raise FileNotFoundError(msg)


def plot_stuff(w, J, m1, m2, xl='w [cm^-1]', yl='J [arb. units]', save_plot=False):
    """
    function to plot takes:
    w - list of np-vectors containing x-values (energy, in cm-1) for each initial condition
    J - list of np-vectors containing y-values (J) for each initial condition
    xl - string containing the label on the x-axis
    yl - string containing the label on the y-axis
    save_plot - bool describing if plot should be saved as .pdf or not

    does:
    1) ask user for a maximum for the x-axis
    2) prepares the graph for plotting, including labels etc
    3) if wanted, saves plot
    4) shows the plot
    """

    question = "What is the maximal value of w (in cm^-1) that you want \
    to plot (float/int)? [Default: highest value] "
    
    maxw = ask_question(question, special='float', default=w[0][-1])
    filename = "SpecDens_{:d}_{:d}.pdf".format(m1, m2)
    with PdfPages(filename) as pdf:
        for i, j  in enumerate(J):
            plt.plot(w[i], j, label='icond{:d}'.format(i))
            plt.xlabel(xl)
            plt.ylabel(yl)
            plt.xlim(w[0][0], maxw)
            plt.legend()
        if save_plot:
            pdf.savefig()
            d = pdf.infodict()
            d['Title'] = 'Spectral Density Figure '
            d['Author'] = 'Autmotically created'
            d['Subject'] = 'SpecDens PDF'
            d['Keywords'] = 'spectral density'
            d['CreationDate'] = datetime.datetime.today()
            d['ModDate'] = datetime.datetime.today()
        plt.show()


def main():
    i1, i2 = ask_for_states()
    question = "Do you want to save the plot as SpecDens_{}_{}.pdf (y/n)? \
    [Default: n] ".format(i1, i2)
    save_fig = ask_question(question, special='bool', default='n')
    w, J = read_files(i1, i2)
    plot_stuff(w, J, i1, i2, save_plot=save_fig)


# ============<>===============
if __name__ == "__main__":
    main()
