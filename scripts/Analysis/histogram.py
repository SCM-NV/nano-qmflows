import matplotlib
matplotlib.use('Agg')

from os.path import join

import argparse
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import os

#  ======================================<>====================================
msg = " script -od output dir -md macrodir -s1 state1 -s2 state2"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-od', required=True,
                    help='Path to the output folder')
parser.add_argument('-s1', required=True, help='State 1', type=int)
parser.add_argument('-s2', required=True, help='State 2', type=int)


def read_cmd_line():
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    out_dir = args.od
    s1 = args.s1
    s2 = args.s2

    return out_dir, s1, s2

#  ======================================<>====================================
ry2ev = 13.60569172


def parse_energies(filePath):
    """
    returns a matrix contaning the energies for each time in each row.
    """
    with open(filePath, 'r') as f:
        xss = f.readlines()
    rss = [[float(x) for i, x in enumerate(l.split())
            if i % 2 == 1 and i > 4] for l in xss]

    return np.array(rss)


def parse_population(filePath):
    """
    returns a matrix contaning the pop for each time in each row.
    """
    with open(filePath, 'r') as f:
        xss = f.readlines()
    rss = [[float(x) for i, x in enumerate(l.split())
            if i % 2 == 1 and i > 2] for l in xss]

    return np.array(rss)


#  ======================================<>====================================


def read_energies_and_pops_from_out(out_dir):
    """
    Read the energy and population created by PYXAID.
    """
    files_out = os.listdir(out_dir)
    names_out_es, names_out_pop = [fnmatch.filter(files_out, x) for x
                                   in ["*energies*", "out*"]]
    paths_out_es, paths_out_pop = [[join(out_dir, x) for x in xs]
                                   for xs in [names_out_es, names_out_pop]]
    ess = list(map(parse_energies, paths_out_es))
    pss = map(parse_population, paths_out_pop)

    return ess, pss


def calculate_band_gap_histo(arr, bins=50):
    """
    Generate a distribution of the Energy gap of two states during the MD
    simulation.
    """
    xs, ys = np.histogram(arr, bins=bins)
    rs = np.empty(bins)
    sh, = arr.shape
    for i in range(bins):
        rs[i] = (ys[i] + ys[i + 1]) * 0.5
    return xs / sh, rs


def main():
    out_dir, state1, state2 = read_cmd_line()

    # Read output files
    ess = read_energies_and_pops_from_out(out_dir)[0]

    # Make a 3D stack of arrays then calculate the mean value
    # for the same time
    average_es = np.mean(np.stack(ess), axis=0)

    # Energy difference
    xss = average_es[:, state1]
    yss = average_es[:, state2]
    arr = xss - yss

    mean_values, gap_distribution = calculate_band_gap_histo(arr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(gap_distribution, mean_values, 'b-')
    plt.tick_params('both', direction='out')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Probability')
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    plt.savefig('Histogram.png', dpi=300, format='png')

# =================<>================================
if __name__ == "__main__":
    main()
