import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from os.path import join

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

#  ======================================<>====================================
msg = " script -p ProjectName -f5 <path/to/hdf5>"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-p', required=True,
                    help='Project Name')
parser.add_argument('-f5', required=True,
                    help='Path to the HDF5')
parser.add_argument('-nh', help='Number of HOMOS to plot (default 10)')
parser.add_argument('-nl', help='Number of LUMOS to plot (default 10)')


def read_cmd_line():
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    project = args.p
    f5 = args.f5
    nh = args.nh if args.nh is not None else 10
    nl = args.nl if args.nl is not None else 10

    return project, f5, nh, nl


#  ======================================<>====================================
h2ev = 27.2114  # hartrees to electronvolts


def fetch_data(project, path_HDF5):
    """
    Get the eigenvalues store in a HDF5 file, under the project root.
    :param project: Projects name
    :type project: String
    :param path_HDF5: Path to the HDF5 file that contains the
    numerical results.
    :type path_HDF5: String
    :returns: list of numpy arrays containing the eigenvalues in ev
    """
    with h5py.File(path_HDF5, 'r') as f5:
        dset = f5[project]
        xs = list(filter(lambda x: 'point_' in x, dset.keys()))
        sh = len(xs)
        points = map(lambda x: join(project, 'point_{}'.format(x),
                                    'cp2k/mo/eigenvalues'), range(sh))
        ess  = list(map(lambda x: f5[x].value, points))

    return list(map(lambda x: x.dot(h2ev), ess))


def plot_data(project, pathHDF5, nHOMOS=None, nLUMOS=None):
    """
    Generates a PDF containing the representantion of the eigenvalues for
    a molecular system called `project` and stored in `pathHDF5`.
    """
    ess = fetch_data(project, pathHDF5)
    rs = np.transpose(np.stack(ess))
    ts = np.arange(len(ess))
    with PdfPages('Eigenvalues.pdf') as pp:
        plt.figure(1)
        plt.title('EigenValues')
        plt.ylabel('Energy [ev]')
        plt.xlabel('Time [fs]')
        for i in range(nHOMOS):
            plt.plot(ts, rs[99 - i], 'b')
        for i in range(nLUMOS):
            plt.plot(ts, rs[100 + i], 'g')
        pp.savefig()


def main():
    project, f5, nh, nl = read_cmd_line()
    plot_data(project, f5, nh, nl)

        
        
# =================<>================================

if __name__ == "__main__":
    main()
