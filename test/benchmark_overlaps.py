from functools import partial
from nac.common import (
    Matrix, change_mol_units, getmass, retrieve_hdf5_data, triang2mtx)
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from nac.workflows.initialization import initialize
from os.path import join
from qmworks.parsers import parse_string_xyz
from scipy import sparse
from typing import (List, Tuple)

import h5py
import numpy as np
import os
import shutil


basisname = 'DZVP-MOLOPT-SR-GTH'
path_traj_xyz = 'test/test_files/Cd33Se33_fivePoints.xyz'
scratch_path = 'scratch'
path_original_hdf5 = 'test/test_files/Cd33Se33.hdf5'
path_test_hdf5 = join(scratch_path, 'test.hdf5')
project_name = 'Cd33Se33'


def main():
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)
    try:
        shutil.copy('test/test_files/BASIS_MOLOPT', scratch_path)
        shutil.copy('test/test_files/GTH_POTENTIALS', scratch_path)

        # Run the actual test
        copy_basis_and_orbitals(path_original_hdf5, path_test_hdf5,
                                project_name)

        config = initialize(
            project_name, path_traj_xyz,
            basisname=basisname, path_basis=None,
            path_potential=None, enumerate_from=0,
            calculate_guesses='first', path_hdf5=path_test_hdf5,
            scratch_path=scratch_path)

        # If the MO orbitals are given in Spherical Coordinates transform then to
        # Cartesian Coordinates.
        trans_mtx = retrieve_hdf5_data(path_test_hdf5, config['hdf5_trans_mtx'])
        dictCGFs = config['dictCGFs']

        # Molecular geometries
        geometries = config['geometries']
        molecule_at_t0 = change_mol_units(parse_string_xyz(geometries[0]))

        # Origin of the dipole
        rc = compute_center_of_mass(molecule_at_t0)
        mtx_integrals_spher = calcDipoleCGFS(molecule_at_t0, dictCGFs, rc, trans_mtx)

        print(tuple(map(lambda mtx: mtx.shape, mtx_integrals_spher)))
    finally:
        # remove tmp data and clean global config
        shutil.rmtree(scratch_path)


# @profile
def calcDipoleCGFS(
        atoms: List, cgfsN: List, rc: Tuple, trans_mtx: Matrix) -> Matrix:
    """
    """
    # x,y,z exponents value for the dipole
    exponents = [{'e': 1, 'f': 0, 'g': 0}, {'e': 0, 'f': 1, 'g': 0},
                 {'e': 0, 'f': 0, 'g': 1}]

    dimCart = trans_mtx.shape[1]

    # Partial application
    partial_multipole = partial(calcMtxMultipoleP, atoms, cgfsN, rc)

    # mtx_integrals_triang = tuple(calcMtxMultipoleP(atoms, cgfsN, rc, **kw)
    mtx_integrals_triang = tuple(partial_multipole(**kw)
                                 for kw in exponents)
    mtx_integrals_cart = tuple(triang2mtx(xs, dimCart)
                               for xs in mtx_integrals_triang)
    return tuple(transform2Spherical(trans_mtx, x) for x
                 in mtx_integrals_cart)


def transform2Spherical(trans_mtx: Matrix, matrix: Matrix) -> Matrix:
    """
    Transform from spherical to cartesians using the sparse representation
    """
    trans_mtx = sparse.csr_matrix(trans_mtx)
    transpose = trans_mtx.transpose()

    return trans_mtx.dot(sparse.csr_matrix.dot(matrix, transpose))


def compute_center_of_mass(atoms: List) -> Tuple:
    """
    Compute the center of mass of a molecule
    """
    # Get the masses of the atoms
    symbols = map(lambda at: at.symbol, atoms)
    masses = np.array([getmass(s) for s in symbols])
    total_mass = np.sum(masses)

    # Multiple the mass by the coordinates
    mrs = [getmass(at.symbol) * np.array(at.xyz) for at in atoms]
    xs = np.sum(mrs, axis=0)

    # Center of mass
    cm = xs / total_mass

    return tuple(cm)


def copy_basis_and_orbitals(source, dest, project_name):
    """
    Copy the Orbitals and the basis set from one the HDF5 to another
    """
    keys = [project_name, 'cp2k']
    excluded = ['coupling', 'overlaps', 'swaps']
    with h5py.File(source, 'r') as f5, h5py.File(dest, 'w') as g5:
        for k in keys:
            if k not in g5:
                g5.create_group(k)
            for l in f5[k].keys():
                if not any(x in l for x in excluded):
                    path = join(k, l)
                    f5.copy(path, g5[k])


if __name__ == "__main__":
    main()
