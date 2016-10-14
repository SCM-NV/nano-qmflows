import argparse
import matplotlib
matplotlib.use('Agg')

from os.path import join

import h5py
import matplotlib.pyplot as plt
import numpy as np

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


def plot_data(project, pathHDF5, homo, nHOMOS, nLUMOS, y_lower, y_upper):
    """
    Generates a PDF containing the representantion of the eigenvalues for
    a molecular system called `project` and stored in `pathHDF5`.
    """
    ess = fetch_data(project, pathHDF5)
    rs = np.transpose(np.stack(ess))
    ts = np.arange(len(ess))

    magnifying_factor = 1
    cm2inch = 0.393700787
    size_x = 8.25 * cm2inch * magnifying_factor
    size_y = 6 * cm2inch * magnifying_factor
    fig = plt.figure(figsize=(size_x, size_y), dpi=300 / magnifying_factor)
    ax = fig.add_subplot(111)
    plt.title('EigenValues')
    plt.ylabel('Energy [ev]')
    plt.xlabel('Time [fs]')
    if y_lower is not None and y_upper is not None:
        plt.ylim(y_lower, y_upper)
    for i in range(nHOMOS):
        plt.plot(ts, rs[homo - i], 'b')
    for i in range(nLUMOS):
        plt.plot(ts, rs[homo + 1 + i], 'g')
        plt.tight_layout()
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    plt.savefig('Eigenvalues.png', dpi=300 / magnifying_factor, format='png')

    plt.show()


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['p', 'hdf5', 'homo', 'nh', 'nl', 'yl', 'yu']
    
    return [getattr(args, p) for p in attributes]


if __name__ == "__main__":
    msg = " script -p project_name -hdf5 <path/to/hdf5> [-homo n -nh nh -nl nl -yl float -yu float]"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True, help='Project name')
    parser.add_argument('-hdf5', required=True, help='path to the HDF5 file')
    parser.add_argument('-homo', help='homo index', type=int, default=19)
    parser.add_argument('-nh', help='Number of HOMOS', type=int, default=10)
    parser.add_argument('-nl', help='Number of LUMOS', type=int, default=10)
    parser.add_argument('-yl', help='Lower limit of y-axis (ev)', type=float, default=-6)
    parser.add_argument('-yu', help='upper limit of y-axis (ev)', type=float, default=1)

    plot_data(*read_cmd_line(parser))
