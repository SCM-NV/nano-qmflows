__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from itertools import chain
from os.path import join

import h5py
import numpy as np
# ==================> Internal modules <==========
from nac.integrals import (calculateCoupling3Points,
                           compute_overlaps_for_coupling,
                           correct_phases)
from nac.common import (femtosec2au, retrieve_hdf5_data)
from qmworks.hdf5.quantumHDF5 import StoreasHDF5

# Types hint
from typing import (Dict, List, Tuple)

# Numpy type hints
Vector = np.ndarray
Matrix = np.ndarray

# ==============================> Schedule Tasks <=============================


def lazy_couplings(paths_overlaps: List, path_hdf5: str, project_name: str,
                   enumerate_from: int, dt: float) -> List:
    """
    :parameter dt: dynamic integration time
    :type      dt: Float (Femtoseconds)
    """
    # time in atomic units
    dt_au = dt * femtosec2au

    # Compute the dimension of the coupling matrix
    mtx_0 = retrieve_hdf5_data(path_hdf5, paths_overlaps[0][0])
    _, dim = mtx_0.shape

    # Read all the Overlaps
    concat_paths = chain(*paths_overlaps)
    overlaps = np.stack([retrieve_hdf5_data(path_hdf5, ps)
                         for ps in concat_paths])

    # Number of couplings to compute
    nCouplings = overlaps.shape[0] // 2 - 1

    # Compute all the phases
    mtx_phases = compute_phases(overlaps, nCouplings, dim)

    # Compute the couplings using the four matrices previously calculated
    # Together with the phases
    paths_couplings = []

    for i in range(nCouplings):
        # Path were the couplinp is store
        k = i + enumerate_from
        path = join(project_name, 'coupling_{}'.format(k))
        with h5py.File(path_hdf5, 'r+') as f5:
            is_done  = path in f5

        # Skip the computation if the coupling is already done
        if is_done:
            print("Coupling: ", path, " has already been calculated")
        else:
            print("Computing coupling: ", path)
            # Tensor containing the overlaps
            j = 2 * i
            ps = overlaps[j: j + 4, :, :]
            # Correct the Phase of the Molecular orbitals
            fixed_phase_overlaps = correct_phases(
                ps, mtx_phases[i: i + 3, :], dim)

            # Compute the couplings with the phase corrected overlaps
            couplings = calculateCoupling3Points(dt_au, *fixed_phase_overlaps)

            with h5py.File(path_hdf5, 'r+') as f5:
                store = StoreasHDF5(f5, 'cp2k')
                store.funHDF5(path, couplings)

        paths_couplings.append(path)

    return paths_couplings


def compute_phases(overlaps: List, nCouplings: int, dim: int) -> Matrix:
    """
    Compute the phase of the state_i at time t + dt, using the following
    equation:
    phase_i(t+dt) = Sii(t) * phase_i(t)
    """
    # initial references
    references = np.ones(dim)

    # Matrix containing the phases
    mtx_phases = np.empty((nCouplings + 2, dim))
    mtx_phases[0, :] = references

    # Compute the phases at times t + dt using the phases at time t
    for i in range(nCouplings + 1):
        Sji_t = overlaps[2 * i, :, :].reshape(dim, dim)
        phases = np.sign(np.diag(Sji_t)) * references
        mtx_phases[i + 1] = phases
        references = phases

    return mtx_phases


def lazy_overlaps(i: int, project_name: str, path_hdf5: str, dictCGFs: Dict,
                  geometries: Tuple, mo_paths: List, hdf5_trans_mtx: str=None,
                  enumerate_from: int=0, nHOMO: int=None,
                  couplings_range: Tuple=None) -> str:
    """
    Calculate the 4 overlap matrix used to compute the subsequent couplings.
    The overlap matrices are computed using 3 consecutive set of MOs and
    3 consecutive geometries( in atomic units), from a molecular dynamics.

    :param i: nth coupling calculation
    :type i: int
    :param project_name: Name of the project to be executed.
    :type project_name: str
    :paramter dictCGFS: Dictionary from Atomic Label to basis set
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :parameter geometries: molecular geometries stored as list of
                           namedtuples.
    :type      geometries: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :parameter mo_paths: List of paths to the MO in the HDF5
    :type      mo_paths: [str]
    :param hdf5_trans_mtx: path to the transformation matrix in the HDF5 file.
    :type hdf5_trans_mtx: str
    :param enumerate_from: Number from where to start enumerating the folders
    create for each point in the MD
    :type enumerate_from: int
    :param nHOMO: index of the HOMO orbital in the HDF5
    :param couplings_range: range of Molecular orbitals used to compute the
    coupling.

    :returns: path to the Coupling inside the HDF5
    """
    # Path inside the HDF5 where the overlaps are stored
    root = join(project_name, 'overlaps_{}'.format(i + enumerate_from))
    names_matrices = ['mtx_sji_t0', 'mtx_sij_t0']
    overlaps_paths_hdf5 = [join(root, name) for name in names_matrices]

    # Test if the overlap is store in the HDF5 calculate it
    with h5py.File(path_hdf5, 'r') as f5:
        is_done = all(path in f5 for path in overlaps_paths_hdf5)

    # If the Overlaps are not in the HDF5 file compute them
    if is_done:
        print(overlaps_paths_hdf5, " Overlaps are already in the HDF5")
    else:
        # Read the Molecular orbitals from the HDF5
        print("Computing: ", root)

        # Paths to the MOs inside the HDF5
        hdf5_mos_path = [mo_paths[i + j][1] for j in range(2)]

        # Partial application of the function computing the overlap
        overlaps = compute_overlaps_for_coupling(
            geometries, path_hdf5, hdf5_mos_path, dictCGFs, nHOMO,
            couplings_range, hdf5_trans_mtx)

        # Store the matrices in the HDF5 file
        with h5py.File(path_hdf5, 'r+') as f5:
            store = StoreasHDF5(f5, 'cp2k')
            for p, mtx in zip(overlaps_paths_hdf5, overlaps):
                store.funHDF5(p, mtx)

    return overlaps_paths_hdf5


def write_hamiltonians(path_hdf5: str, mo_paths: List, path_couplings: List,
                       nPoints: int, path_dir_results: str=None,
                       enumerate_from: int=0, nHOMO: int=None,
                       couplings_range: Tuple=None) -> None:
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
        j = i + enumerate_from
        path_coupling = path_couplings[i]
        css = retrieve_hdf5_data(path_hdf5, path_coupling)

        # Extract the energy values
        energies = retrieve_hdf5_data(path_hdf5, mo_paths[i][0])
        if all(x is not None for x in [nHOMO, couplings_range]):
            lowest = nHOMO - couplings_range[0]
            highest = nHOMO + couplings_range[1]
            energies = energies[lowest: highest]

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
