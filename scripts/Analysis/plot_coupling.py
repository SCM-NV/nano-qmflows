#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob

from interactive import ask_question


def read_value(fname, mo1=101, mo2=100):
    """
    function that reads 1 value (at mo1,mo2) from 1 file and returns it as a float
    """
    xs = np.loadtxt(fname)
    try:
        return abs(xs[mo1, mo2])
    except IndexError:
        print('IndexError; you probably provided a non-existing MO-integer?')


def plot_stuff(arr, mo1, mo2, xl='time [fs]', yl='n.a. coupling [meV]',
               plot_mean=True, save_plot=False):
    """
    function that takes
    arr - a vector of y-values that are plot
    mo1 - int describing first MO
    mo2 - int describing second MO
    xl,yl - strings describing x and y axis labels
    plot_mean, save_plot - bools telling to plot the mean and save the plot or not, respectively
    """

    magnifying_factor = 2
    cm2inch = 0.393700787
    label = 'Coupling_{:d}_{:d}'.format(mo1, mo2)
    plt.figure(figsize=(8.25 * cm2inch * magnifying_factor,
                        6 * cm2inch * magnifying_factor))
    plt.plot(arr, c='k', label=label)
    plt.xlabel(xl)
    plt.ylabel(yl)
    mean_val = np.mean(np.absolute(arr))

    print('The average of the absolute coupling is {0:.2f} meV'.format(mean_val))

    if plot_mean:
        plt.plot([mean_val] * len(arr), color='#999999', label='Average absolute coupling')
    plt.legend()
    plt.tight_layout()

    if save_plot:
        fileName  = "coupling_{:d}_{:d}.pdf".format(mo1, mo2)
        plt.savefig(fileName, format='pdf', dpi=300 / magnifying_factor)
    plt.show()


def compute_values(fn, m1, m2, r2meV):
    r = read_value(fn, mo1=m1, mo2=m2)
    return r * r2meV


def main():
    r2meV = 13605.698  # conversion from rydberg to meV

    print("The labeling is as\n...\n98 HOMO-1\n99 HOMO\n100 LUMO\n101 LUMO+1\n...")
    m1 = ask_question("Please define the first MO (integer). [Default: 101] ",
                      special='int', default='101')
    m2 = ask_question("Please define the second MO (integer). [Default: 100] ",
                      special='int', default='100')
    q3 = ask_question(("Do you want to plot a line of the average value in the"
                       " graph (y/n) ? [Default: y] "), special='bool',
                      default='y')
    p4 = ("Do you want to save the graph as coupling_{:d}_{:d}.pdf (y/n) ? "
          "[Default: n]".format(m1, m2))
    q4 = ask_question(p4, special='bool', default='n')

    files = glob.glob('Ham_*_im')

    if files:
        list_coupling = [compute_values('Ham_{}_im'.format(i), m1, m2, r2meV)
                         for i in range(len(files))]
        plot_stuff(list_coupling, m1, m2, plot_mean=q3, save_plot=q4)
    else:
        print('ERROR: No files found. Please make sure that you are in the '
              'hamiltonians directory.')


# ============<>===============
if __name__ == "__main__":
    main()
