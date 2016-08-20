#! /usr/bin/env python

import matplotlib.pyplot as plt
import fnmatch
import numpy as np
import os
from interactive import ask_question


msg = ('This is a program that plots the decorense for a certain'
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


def read_spec_files(i1, i2):
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


def read_files(i1, i2, name=False):
    """
    function that opens all the dephasing files of all the initial
    conditions for states i1 and i2
    and returns
    t - a list of np vectors (describing time) for all initial conditions.
    d - list of np-vectors (describing D) for each initial condition.
    fd - list of np-vectors (describing fitted D) for each initial condition.
    naf - list of np-vectors (describing norm. autocorrelation function) for
    each initial condition.
    uaf - list of np-vectors (describing unnorm. autocorrelation function)
    for each initial condition.
    sc - list of np-vectors (describing second cumulant) for each
    initial condition.
    """
    t, d, fd, naf, uaf, sc = [], [], [], [], [], []
    files = os.listdir('.')
    name = 'icond*pair{:d}_{:d}Dephasing_function.txt'.format(i1, i2)
    density_files = fnmatch.filter(files, name)
    if density_files:
        for i, filename in enumerate(density_files):
            arr = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4, 5), skiprows=1)
            arr = np.transpose(arr)
            t.append(arr[0])
            d.append(arr[1])
            fd.append(arr[2])
            naf.append(arr[3])
            uaf.append(arr[4])
            sc.append(arr[5])
        return t, d, fd, naf, uaf, sc
    else:
        name2 = name[0:4] + '0' + name[6:]
        msg = ('File not found.\nAre you in the out folder? And are '
               'you sure the ints are correct?\n'
               'The program was looking for a file named: \'{}\' in your '
               'current directory.'.format(name2))
        raise FileNotFoundError(msg)


def plot_stuff(t, d, fd, naf, uaf, sc, w, J, m1, m2, save_plot=False, plot_avg=True):
    """
    function to plot
    
    takes:
    :param t: list of np-vectors containing x-values (time, usually(!) in fs).
    for each initial condition.
    :param d: list of np-vectors containing y-values (D) for each initial
    condition.
    :param fd: list of np-vectors containing y-values (fitted D) for each
    initial condition.
    :param naf: list of np-vectors containing y-values.
    (norm. autocorrelation function) for each initial condition
    :param uaf: list of np-vectors containing y-values
    (unnorm. autocorrelation function) for each initial condition.
    :param sc: list of np-vectors containing y-values (second cumulant) for each
    initial condition.
    :param yl: string containing the label on the y-axis.
    :param save_plot: bool describing if plot should be saved as .pdf or not.

    does:
    1) creates 2x2 'grid' within plot
    2.1) adds d, df to subplot1,1
    2.2) adds naf and uaf to subplot1,2
    2.3) adds sc to subplot2,1
    2.4) adds J to subplot2,2
    3) if wanted, saves plot
    4) shows the plot
    """
    magnifying_factor = 2

    # 1
    cm2inch = 0.393700787
    size_x = 16.5 * cm2inch * magnifying_factor
    size_y = 12.3 * cm2inch * magnifying_factor
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(size_x, size_y),
                                               sharex=False, sharey=False)
    colorlist = ['r', 'b', 'g', 'y', 'k', 'm']
    question = "What is the maximal value of w (in cm^-1) that you want \
    to plot (x-axis spectral density plot) (type: float/int)? [Default: highest value] "

    maxw = ask_question(question, special='float', default=w[0][-1])
    conv_fac = uaf[0][0] / naf[0][0]
    shape = len(d)
    # 2.1
    for i in range(shape):
        ax1.plot(t[i], d[i], color=colorlist[i], label='D_{:d}'.format(i))
        ax1.plot(t[i], fd[i], color='#999999', label='fitD_{:d}'.format(i))
    ax1.set_ylabel('D [arb. units]')
    ax1.set_xlabel('time [fs]')
    ax1.set_xlim(0, 200)

    # 2.2
    for i in range(shape):
        ax2.plot(t[i], naf[i], color=colorlist[i], label='norm_autocorr_{:d}'.format(i))
    if plot_avg:
        ax2.plot(t[0], [np.mean(naf[0])] * len(naf[0]), color='#999999', label='average')
    ax2.axhline(y=0, color='black')
    ax2.set_ylabel('NAF [arb. units]')
    ax2.set_xlabel('time [fs]')
    ax5 = ax2.twinx()
    ax5.set_ylim(ax2.get_ylim())
    ax5.set_ylabel('UAF [arb. units]')
    labels = ax2.yaxis.get_ticklocs()
    labels2 = []
    for label in labels:
        labels2.append('{:.2e}'.format(label * conv_fac))
    ax5.set_yticklabels(labels2)

    # 2.3
    for i in range(shape):
        ax3.plot(t[i], sc[i], color=colorlist[i], label='scnd_cumul_{:d}'.format(i))
    # if plot_avg:
    #     ax3.plot(t[0],[np.mean(sc)]*len(sc[0]),color='#999999', label='average')
    ax3.set_xlabel('time [fs]')
    ax3.set_ylabel('SC [arb. units]')

    # 2.4
    for i in range(len(d)):
        ax4.plot(w[i], J[i], color=colorlist[i], label='icond{:d}'.format(i))
    if plot_avg:
        ax4.plot(w[0], [np.mean(J)] * len(J[0]), color='#999999', label='average')
    ax4.set_xlabel('w [cm^-1]')
    ax4.set_ylabel('J [arb. units]')
    ax4.set_xlim(w[0][0], maxw)
    plt.tight_layout()

    # 3
    if save_plot:
        filename = "Dec_{:d}_{:d}.pdf".format(m1, m2)
        plt.savefig(filename, dpi=300 / magnifying_factor, format='pdf')

    # 4
    plt.show()


def main():
    i1, i2 = ask_for_states()
    question = "Do you want to save the plot as Dec_{}_{}.pdf (y/n)? \
    [Default: n] ".format(i1, i2)
    save_fig = ask_question(question, special='bool', default='n')
    question2 = 'Do you want to plot the average (y/n)? [Default: y]'
    plot_avg = ask_question(question2, special='bool', default='y')
    w, J = read_spec_files(i1, i2)
    t, d, fd, naf, uaf, sc = read_files(i1, i2)
    plot_stuff(t, d, fd, naf, uaf, sc, w, J, i1, i2, save_plot=save_fig,
               plot_avg=plot_avg)


# ============<>===============
if __name__ == "__main__":
    main()
