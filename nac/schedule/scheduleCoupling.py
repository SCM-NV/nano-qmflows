__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from noodles import schedule  # Workflow Engine
from os.path import join

import h5py
import numpy as np
# ==================> Internal modules <==========
from nac.integrals import calculateCoupling3Points
from nac.common import (femtosec2au, retrieve_hdf5_data)
from qmworks.hdf5.quantumHDF5 import StoreasHDF5

# ==============================> Schedule Tasks <=============================


@schedule
def lazy_schedule_couplings(i, path_hdf5, dictCGFs, geometries, mo_paths, dt=1,
                            hdf5_trans_mtx=None, output_folder=None,
                            enumerate_from=0, nCouplings=None):
    """
    Calculate the non-adiabatic coupling using 3 consecutive set of MOs in
    a dynamics, using 3 consecutive geometries in atomic units.

    :param i: nth coupling calculation
    :type i: int
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :parameter geometries: molecular geometries stored as list of
                           namedtuples.
    :type      geometries: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :parameter coefficients: Tuple of Molecular Orbital coefficients.
    :type      coefficients: (Matrix, Matrix, Matrix)
    :parameter dt: dynamic integration time
    :type      dt: Float (Femtoseconds)
    :param hdf5_trans_mtx: path to the transformation matrix in the HDF5 file.
    :type hdf5_trans_mtx: String
    :param output_folder: name of the path inside the HDF5 where the coupling
    is stored.
    :type output_folder: String
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :type enumerate_from: Int
    :returns: path to the Coupling inside the HDF5
    """
    def calc_coupling(output_path, dt):

        if hdf5_trans_mtx is not None:
            trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx)
        else:
            trans_mtx = None

        mos = tuple(map(lambda j:
                        retrieve_hdf5_data(path_hdf5,
                                           mo_paths[i + j][1]), range(3)))

        # Calculate the coupling among nCouplings/2 HOMOs and  nCouplings/2 LUMOs.
        if nCouplings is not None:
            _, nStates = mos[0].shape
            middle, ncs  = [n // 2 for n in  [nStates, nCouplings]]
            lower, upper = middle - ncs, middle + ncs
            # Extrract a subset of nCouplings coefficients
            mos = tuple(map(lambda xs: xs[:, lower: upper], mos))

        # time in atomic units
        dt_au = dt * femtosec2au
        rs = calculateCoupling3Points(geometries, mos, dictCGFs, dt_au, trans_mtx)

        # Store the couplings
        with h5py.File(path_hdf5) as f5:
            store = StoreasHDF5(f5, 'cp2k')
            store.funHDF5(output_path, rs)

    if output_folder is None:
        msg = 'There was not specified a path in the HDF5 file to store the coupling\n'
        raise RuntimeError(msg)

    output_path = join(output_folder, 'coupling_{}'.format(i + enumerate_from))
    print("Calculating Coupling: ", output_path)
    # Test if the coupling is store in the HDF5 calculate it
    with h5py.File(path_hdf5, 'r') as f5:
        is_done = output_path in f5
    if not is_done:
        calc_coupling(output_path, dt)

    return output_path


def write_hamiltonians(path_hdf5, work_dir, mo_paths, path_couplings, nPoints,
                       path_dir_results=None, enumerate_from=0,
                       nCouplings=None):
    """
    Write the real and imaginary components of the hamiltonian using both
    the orbitals energies and the derivative coupling accoring to:
    http://pubs.acs.org/doi/abs/10.1021/ct400641n
    **Units are: Rydbergs**.
    """
    def write_pyxaid_format(arr, fileName):
        np.savetxt(fileName, arr, fmt='%10.5e', delimiter='  ')

    def energySubset(xs):
        if nCouplings is None:
            return xs
        else:
            dim, = xs.shape
            middle = dim // 2
            ncs = nCouplings // 2
            return xs[middle - ncs: middle - ncs]

    ham_files = []
    for i in range(nPoints):
        path_coupling = path_couplings[i]
        css = retrieve_hdf5_data(path_hdf5, path_coupling)
        energies = energySubset(retrieve_hdf5_data(path_hdf5, mo_paths[i][0]))
        j = i + enumerate_from

        # FileNames
        file_ham_im = join(path_dir_results, 'Ham_{}_im'.format(j))
        file_ham_re = join(path_dir_results, 'Ham_{}_re'.format(j))

        # convert to Rydbergs
        ham_im = 2.0 * css
        ham_re = np.diag(2.0 * energies)

        write_pyxaid_format(ham_im, file_ham_im)
        write_pyxaid_format(ham_re, file_ham_re)
        ham_files.append((file_ham_im, file_ham_re))

    return ham_files
