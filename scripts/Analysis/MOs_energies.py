from os.path import join
from interactive import ask_question

import h5py
import matplotlib.pyplot as plt
import numpy as np

#  ======================================<>====================================
def obtain_data():
    project = ask_question('What is the project name? ')
    f5 = ask_question('What is the path of the hdf5-file? ')
    nh = ask_question('What is the number of HOMOs to plot? [Default: 10] ', special='int', default='10')
    nl = ask_question('What is the number of LUMOs to plot? [Default: 10] ', special='int', default='10')
    save_fig = ask_question('Do you want to save the plot (y/n)? [Default: n] ', special='bool', default='n')
    return project, f5, nh, nl, save_fig


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


def plot_data(project, pathHDF5, nHOMOS, nLUMOS, save_fig):
    """
    Generates a PDF containing the representantion of the eigenvalues for
    a molecular system called `project` and stored in `pathHDF5`.
    """
    ess = fetch_data(project, pathHDF5)
    rs = np.transpose(np.stack(ess))
    ts = np.arange(len(ess))


    magnifying_factor = 1
    cm2inch = 0.393700787
    plt.figure(figsize=(8.25*cm2inch*magnifying_factor, 6*cm2inch*magnifying_factor), dpi= 300/magnifying_factor )
    plt.title('EigenValues')
    plt.ylabel('Energy [ev]')
    plt.xlabel('Time [fs]')
    for i in range(nHOMOS):
        plt.plot(ts, rs[99 - i], 'b')
    for i in range(nLUMOS):
        plt.plot(ts, rs[100 + i], 'g')

        plt.tight_layout()

    if save_fig:
        plt.savefig('Eigenvalues.pdf', dpi=300 / magnifying_factor, format='pdf')

    plt.show()


def main():
    project, f5, nh, nl, save_fig = obtain_data()
    print(f5)
    plot_data(project, f5, nh, nl, save_fig)

# =================<>================================

if __name__ == "__main__":
    main()
