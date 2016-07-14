import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from os.path import join

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

#  ===================<>===================                                                                            


def fetch_data(project, path_HDF5):
    with h5py.File(path_HDF5, 'r') as f5:
        dset = f5[project]
        xs = filter(lambda x: 'point_' in x, dset.keys())
        points = map(lambda x: join(project, x ,'cp2k/mo/eigenvalues'), xs)
        ess  = list(map(lambda x: f5[x].value, points))
        print([x.min() for x in ess])

    return ess

def main():
    path_HDF5 = '/home/felipe/WorkBench_Python/nonAdiabaticCoupling/test/test_files/pentacene_test.hdf5'
    project = 'nfs/home6/fza900/pentacene/test/test_pentacene'
    # path = join('/scratch-shared/fza900' , project, 'quantum.hdf5')
    ess = fetch_data(project, path_HDF5)
    ts = np.arange(len(ess))
    with PdfPages('Energies.pdf') as pp:
        plt.figure(1)
        plt.title('EigenValues')
        plt.plot(ts, ess)
        plt.ylabel('Energy [hartree]')
        plt.xlabel('Time [fs]')
        pp.savefig()

# =================<>================================                                                                  
if __name__ == "__main__":
    main()
