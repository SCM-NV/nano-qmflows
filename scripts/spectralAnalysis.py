#! /usr/bin/env python

from matplotlib.backends.backend_pdf import PdfPages

import datetime
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


msg = """This is a program that plots the spectral density for a certain
pair of states.Usage: Make sure that you are in the out folder containing the
icond-files and fill in the prompted questions."""


def ask_question(q_str, special=None, default=None):
    """
    function that ask the question, and parses the answer to prefered format
    q_str = string containing the question to be asked
    returns string, bool (spec='bool'), int (spec='int') or float (spec='float')
    """
    done = False
    while True:
        if sys.version_info[0] == 3:
            question = str(input(q_str))
        elif sys.version_info[0] == 2:
            question = str(raw_input(q_str))
        if special is None:
            return question
        elif special == 'bool':
            if question == 'y' or question == 'yes':
                return True
            elif question == 'n' or question == 'no':
                return False
            else:
                return default
        elif special == 'float':
            done = True
            a = 0.
            try:
                a = float(question)
            except ValueError:
                if default == None:
                    print("This is not a float/integer? Please try again.")
                    done = False
                else:
                    a = default
            if done:
                return a
        elif special == 'int':
            done = True
            a = 0
            try:
                a = int(question)
            except ValueError:
                if default is None:
                    print("This is not an integer? Please try again.")
                    done = False
                else:
                    a = default
            if done:
                return a


def ask_for_states():
    print("""The states are labeled as in PYXAID; the first state being 0.
    You can look them up in the output.""")
    i1 = ask_question("What is the integer representing the first state (int)?",
                      special='int')
    i2 = ask_question("What is the integer representing the second state (int)?",
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
        for i, filename in enumerate(density_files):
            arr = np.loadtxt(filename, usecols=(3, 5))
            arr = np.transpose(arr)
            w.append(arr[0])
            j.append(arr[1])
        return w, j
    else:
        msg = """File not found.\nAre you in the out folder? And are \
        you sure the ints are correct?\n.\
        The program was looking for a file named: {} in your \
        current directory.""".format(filename)
        raise FileNotFoundError(msg)


def plot_stuff(w, J, m1, m2, xl='w [cm^-1]', yl='J', save_plot=False):
    """
    Document me...
    """
    question = "What is the maximal value of w (in cm^-1) that you want \
    to plot (float/int)? [Default: highest value] "
    
    maxw = ask_question(question, special='float', default=np.max(J[0]))
    filename = "SpecDens_{:d}_{:d}.pdf".forma(m1, m2)
    with PdfPages(filename) as pdf:
        for i in range(len(J)):
            plt.plot(w, J[i], label='icond{:d}'.format(i))
            plt.xlabel(xl)
            plt.ylabel(yl)
            plt.xlim(w[0], maxw)
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
    [Default: n] ".forma(i1, i2)
    save_fig = ask_question(question, special='bool', default=False)
    w, J = read_files(i1, i2)
    plot_stuff(w, J, i1, i2, save_plot=save_fig)


# ============<>===============
if __name__ == "__main__":
    main()
