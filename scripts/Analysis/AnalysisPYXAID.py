import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
from os.path import join

import argparse
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import os

#  ======================================<>====================================
msg = " script -d out"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-od', required=True,
                    help='Path to the output folder')
parser.add_argument('-md', required=True,
                    help='Path to the macro folder')
parser.add_argument('-ho', required=True, help='HOMO number', type=int)
parser.add_argument('-lu', help='LUMO number', type=int)
parser.add_argument('-n', help='Number of chunks to split the trajectory',
                    type=int)


def read_cmd_line():
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    out_dir = args.od
    macro_dir = args.md
    homo = args.ho
    n = args.n if args.n is not None else 25

    return out_dir, macro_dir, homo, n

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


def get_values(iss, arr, row):
    """
    For a given `row` get the values of array `arr` using indexes iss.
    """
    rs = []
    for index in iss:
        i, j = index
        if i < row:
            continue
        elif i > row:
            break
        else:
            rs.append(arr[i, j])
    return rs


def distribution(x, mu=0, sigma=1):
    """
    Normal Gaussian distribution
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * \
        np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

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


def read_energies_and_pops_from_macro(macro_dir):
    """
    Read both SH and SE energy and population from the macro folder.
    """
    files_macro = os.listdir(macro_dir)
    se_sh_es_files = [fnmatch.filter(files_macro, x)[0]
                      for x in ["se_en_*[!_]", "sh_en_*[!_]"]]

    se_sh_pop_files = [fnmatch.filter(files_macro, x)[0]
                       for x in ["se_pop*", "sh_pop*"]]
    se_sh_es_paths = [join(macro_dir, xs) for xs in se_sh_es_files]
    se_sh_pop_paths = [join(macro_dir, xs) for xs in se_sh_pop_files]

    se_ess, sh_ess = [np.loadtxt(f, usecols=(3,)) for f in se_sh_es_paths]
    se_pop, sh_pop = [np.loadtxt(f, usecols=(3,)) for f in se_sh_pop_paths]

    return se_ess, sh_ess, se_pop, sh_pop


def calc_grid_energies(average_pop, nsteps, lumos):
    """
    # Intraband relaxation graphic standard deviation 0.08 eV
    """
    sigma = 0.08
    grid = lambda x: np.random.normal(loc=x, scale=sigma, size=317)

    # Select those probabilities larger than a threshold
    lim = 0.001
    pss = np.select([average_pop > lim], [average_pop])
    iss = np.argwhere(average_pop > lim)
    # Select the non zero population and corresponding energies
    nonzero_pop = [np.take(pss[i], np.nonzero(pss[i])).flatten()
                   for i in range(0, nsteps, 3)]
    nonzero_es = [get_values(iss, lumos, i) for i in range(0, nsteps, 3)]
    mean_mus = [np.dot(p, es) for p, es in zip(nonzero_pop, nonzero_es)]
    xss = [grid(mu) for mu in mean_mus]
    grid_es = np.stack(xss)
    grid_pop = np.stack(distribution(xss[i], mu=mean_mus[i], sigma=sigma)
                        for i in range(len(xss)))

    time = np.arange(0, nsteps, 3)
    _, grid_time = np.meshgrid(time, time)

    return grid_es, grid_pop, grid_time


def calculate_band_gap_histo(homos, bins=50):
    """
    Generate a distribution of the HOMO-LUMO gap during the MD
    simulation.
    """
    xs, ys = np.histogram(homos, bins=bins)
    rs = np.empty(bins)
    sh, = homos.shape
    for i in range(bins):
        rs[i] = (ys[i] + ys[i + 1]) * 0.5
    return xs / sh, rs


def main():
    out_dir, macro_dir, homo_number, lumo_number = read_cmd_line()

    # Read output files
    ess, pss = read_energies_and_pops_from_out(out_dir)

    # Make a 3D stack of arrays then calculate the mean value
    # for the same time
    average_es = np.mean(np.stack(ess), axis=0)
    # average_pop = np.mean(np.stack(pss), axis=0)

    # Create time steps array
    nsteps, _ = average_es.shape
    ts = np.arange(nsteps)

    # HOMO Energies
    homos = average_es[:, homo_number]

    # Read Macro files
    se_ess, sh_ess, se_pop, _ = read_energies_and_pops_from_macro(macro_dir)

    xss = se_ess - homos
    print(xss[:10] * ry2ev)

    mean_values, gap_distribution = calculate_band_gap_histo(homos)

    with PdfPages('Energies.pdf') as pp:
        plt.figure(1)
        plt.title('SE and SH Energy Evolution')
        plt.plot(ts, se_ess, 'r-', label='SE')
        plt.plot(ts, sh_ess, 'b-', label='SH')
        plt.ylabel('Energy [eV]')
        plt.xlabel('Time [fs]')
        plt.legend(loc='best')
        pp.savefig()
        plt.figure(2)
        plt.title('Intraband relaxation')
        plt.plot(ts, sh_ess - homos, 'b-')
        plt.ylabel('Energy [eV]')
        plt.xlabel('Time [fs]')
        pp.savefig()
        plt.figure(3)
        plt.title('Band Gap')
        plt.plot(gap_distribution, mean_values, 'b-')
        plt.xlabel('Energy [eV]')
        plt.ylabel('Probability')
        pp.savefig()

        # plt.figure(5)
        # plt.title('Intraband relaxation of electron energy')
        # p = plt.pcolormesh(grid_time, grid_es, grid_pop, cmap=plt.cm.bwr)
        # plt.ylabel('Energy [eV]')
        # plt.xlabel('Time [fs]')
        # plt.colorbar(p)
        # plt.savefig('ElectronEnergyDist.png')
        # pp.savefig()

    # es_energy = np.stack([ts, sh_ess - lumo_1], axis=1)
    # es_hole = np.stack([ts, sh_ess - homos], axis=1)
    # np.savetxt("electron_energy.out", es_energy, fmt='%.7e')
    # np.savetxt("hole_energy.out", es_hole, fmt='%.7e')


# =================<>================================
if __name__ == "__main__":
    main()
