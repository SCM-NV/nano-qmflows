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

# Types hint
from typing import (Dict, List, Tuple)

# Numpy type hints
Vector = np.ndarray
Matrix = np.ndarray

# ==============================> Schedule Tasks <=============================


@schedule
def lazy_schedule_couplings(i: int, path_hdf5: str, dictCGFs: Dict,
                            geometries: Tuple, mo_paths: List, dt: float=1,
                            hdf5_trans_mtx: str=None, output_folder: str=None,
                            enumerate_from: int=0,
                            nHOMO: int=None,
                            couplings_range: Tuple=None) -> str:
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
    :param nHOMO: index of the HOMO orbital in the HDF5
    :param couplings_range: range of Molecular orbitals used to compute the
    coupling.

    :returns: path to the Coupling inside the HDF5
    """
    def calc_coupling(output_path, nHOMO, couplings_range, dt):

        if hdf5_trans_mtx is not None:
            trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx)
        else:
            trans_mtx = None

        mos = tuple(map(lambda j:
                        retrieve_hdf5_data(path_hdf5,
                                           mo_paths[i + j][1]), range(3)))

        #  Calculate the coupling in the range provided by the user
        _, nStates = mos[0].shape

        # If the user does not define the number of HOMOs and LUMOs
        # assume that the first half of the read MO from the HDF5
        # are HOMOs and the last Half are LUMOs.
        nHOMO = nHOMO if nHOMO is not None else nStates // 2

        # If the couplings_range variable is not define I assume
        # that the number of LUMOs is nStates - nHOMO
        if couplings_range is None:
            couplings_range = (nHOMO, nStates - nHOMO)

        # Define the range of couplings that are going to be compute
        lower = nHOMO - couplings_range[0]
        upper = couplings_range[1]

        # Extract a subset of molecular orbitals to compute the coupling
        mos = tuple(map(lambda xs: xs[:, lower: nHOMO + upper], mos))

        # time in atomic units
        dt_au = dt * femtosec2au
        rs = calculateCoupling3Points(geometries, mos, dictCGFs, dt_au,
                                      trans_mtx)

        # Store the couplings
        with h5py.File(path_hdf5) as f5:
            store = StoreasHDF5(f5, 'cp2k')
            store.funHDF5(output_path, rs)

    if output_folder is None:
        msg = ('There was not specified a path in the HDF5 file to store \
        the coupling\n')
        raise RuntimeError(msg)

    output_path = join(output_folder, 'coupling_{}'.format(i + enumerate_from))
    print("Calculating Coupling: ", output_path)
    # Test if the coupling is store in the HDF5 calculate it
    with h5py.File(path_hdf5, 'r') as f5:
        is_done = output_path in f5
    if not is_done:
        calc_coupling(output_path, nHOMO, couplings_range, dt)
    else:
        print(output_path, " Coupling is already in the HDF5")
    return output_path


def write_hamiltonians(path_hdf5, work_dir, mo_paths, path_couplings, nPoints,
                       path_dir_results=None, enumerate_from=0):
    """
    Write the real and imaginary components of the hamiltonian using both
    the orbitals energies and the derivative coupling accoring to:
    http://pubs.acs.org/doi/abs/10.1021/ct400641n
    **Units are: Rydbergs**.
    """
    def write_pyxaid_format(arr, fileName):
        np.savetxt(fileName, arr, fmt='%10.5e', delimiter='  ')

    ham_files = []
    for i in range(nPoints):
        path_coupling = path_couplings[i]
        css = retrieve_hdf5_data(path_hdf5, path_coupling)
        energies = retrieve_hdf5_data(path_hdf5, mo_paths[i][0])
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
