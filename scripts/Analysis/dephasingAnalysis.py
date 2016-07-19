#! /usr/bin/env python

import matplotlib.pyplot as plt
import datetime
import fnmatch
import numpy as np
import os
import sys


msg = ('This is a program that plots the decorense for a certain'
'pair of states. Usage: Make sure that you are in the out folder containing the'
'icond-files and fill in the prompted questions.')


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
    i1 = ask_question("What is the integer representing the first state (int)? ",
                      special='int')
    i2 = ask_question("What is the integer representing the second state (int)? ",
                      special='int')
    return i1, i2


def read_files(i1, i2, name = False):
    """
    function that opens all the dephasing files of all the initial
    conditions for states i1 and i2
    and returns
    t - a list of np vectors (describing time) for all initial conditions
    d - list of np-vectors (describing D) for each initial condition
    fd - list of np-vectors (describing fitted D) for each initial condition
    naf - list of np-vectors (describing norm. autocorrelation function) for each initial condition
    uaf - list of np-vectors (describing unnorm. autocorrelation function) for each initial condition
    sc - list of np-vectors (describing second cumulant) for each initial condition
    """
    t, d, fd, naf, uaf, sc = [], [], [], [], [], []
    files = os.listdir('.')
    name = 'icond*pair{:d}_{:d}Dephasing_function.txt'.format(i1, i2)
    density_files = fnmatch.filter(files, name)
    if density_files:
        for i, filename in enumerate(density_files):
            arr = np.loadtxt(filename, usecols=(0,1,2,3,4,5),skiprows=1)
            arr = np.transpose(arr)
            t.append(arr[0])
            d.append(arr[1])
            fd.append(arr[2])
            naf.append(arr[3])
            uaf.append(arr[4])
            sc.append(arr[5])
        return t,d,fd,naf,uaf,sc
    else:
        name2 = name[0:4]+'0'+name[6:]
        msg = ('File not found.\nAre you in the out folder? And are '
        'you sure the ints are correct?\n'
        'The program was looking for a file named: \'{}\' in your '
        'current directory.'.format(name2))
        raise FileNotFoundError(msg)


def plot_stuff(t, d, fd, naf, uaf, sc, m1, m2, save_plot=False):
    """
    function to plot
    
    takes:
    t - list of np-vectors containing x-values (time, usually(!) in fs) for each initial condition
    d - list of np-vectors containing y-values (D) for each initial condition
    fd - list of np-vectors containing y-values (fitted D) for each initial condition
    naf - list of np-vectors containing y-values (norm. autocorrelation function) for each initial condition
    uaf - list of np-vectors containing y-values (unnorm. autocorrelation function) for each initial condition
    sc - list of np-vectors containing y-values (second cumulant) for each initial condition
    yl - string containing the label on the y-axis
    save_plot - bool describing if plot should be saved as .pdf or not

    does:
    1) creates 2x2 'grid' within plot
    2.1) adds d, df to subplot1,1
    2.2) adds naf to subplot1,2
    2.3) adds sc to subplot2,1
    2.4) adds uaf to subplot2,2
    3) if wanted, saves plot
    4) shows the plot
    """

    # 1
    cm2inch = 0.393700787
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16.5*cm2inch, 12.3*cm2inch), dpi=300, sharex=False, sharey=False)

    # 2.1
    for i in range(len(d)):
        ax1.plot(t[i], d[i],color='k', label='D_{:d}'.format(i))
        ax1.plot(t[i],fd[i],color='#999999', label='fitD_{:d}'.format(i))
        ax1.set_ylabel('D [arb. units]')
        ax1.set_xlabel('time [fs]')
        ax1.set_xlim(0,200)
        # ax1.set_xscale('log')
        # ax1.set_yscale('log')

    # 2.2
    for i in range(len(d)):
        ax2.plot(t[i], naf[i], label='norm_autocorr_{:d}'.format(i))
        ax2.set_ylabel('NAF [arb. units]')
        ax2.set_xlabel('time [fs]')

    # 2.3
    for i in range(len(d)):
        ax3.plot(t[i], sc[i], label='scnd_cumul_{:d}'.format(i))
        ax3.set_xlabel('time [fs]')
        ax3.set_ylabel('SC [arb. units]')

    # 2.4
    for i in range(len(d)):
        ax4.plot(t[i], uaf[i], label='icond{:d}'.format(i))
        ax4.set_xlabel('time [fs]')
        ax4.set_ylabel('UAF [arb. units]')

    # 3
    if save_plot:
        filename = "Dec_{:d}_{:d}.pdf".format(m1, m2)
        plt.savefig(filename)

    # 4
    plt.show()



def main():
    i1, i2 = ask_for_states()
    question = "Do you want to save the plot as Dec_{}_{}.pdf (y/n)? \
    [Default: n] ".format(i1, i2)
    save_fig = ask_question(question, special='bool', default=False)
    t, d, fd, naf, uaf, sc = read_files(i1, i2)
    plot_stuff(t, d, fd, naf, uaf, sc, i1, i2, save_plot=save_fig)


# ============<>===============
if __name__ == "__main__":
    main()

