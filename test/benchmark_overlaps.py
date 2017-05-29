
from nac.common import (
    Matrix, change_mol_units, getmass, retrieve_hdf5_data, triang2mtx)
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from nac.workflows.initialization import initialize
from os.path import join
from qmworks.parsers import parse_string_xyz
from scipy import sparse

import numpy as np
import os
import shutil


from typing import (List, Tuple)

basisname = 'DZVP-MOLOPT-SR-GTH'
path_traj_xyz = 'Cd33Se33_fivePoints.xyz'
scratch_path = 'scratch'
path_original_hdf5 = 'Cd33Se33.hdf5'
path_test_hdf5 = join(scratch_path, 'test.hdf5')
project_name = 'Cd33Se33'


def main():

    copy_files()
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

    # Contracted Gaussian functions normalized
    cgfsN = [dictCGFs[x.symbol] for x in molecule_at_t0]

    # Origin of the dipole
    rc = compute_center_of_mass(molecule_at_t0)
    mtx_integrals_spher = calcDipoleCGFS(molecule_at_t0, cgfsN, rc, trans_mtx)

    print(tuple(map(lambda mtx: mtx.shape, mtx_integrals_spher)))


def copy_files():
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)
        shutil.copy(path_original_hdf5, path_test_hdf5)


def calcDipoleCGFS(
        atoms: List, cgfsN: List, rc: Tuple, trans_mtx: Matrix) -> Matrix:
    """
    Compute the Multipole matrix in cartesian coordinates and
    expand it to a matrix and finally convert it to spherical coordinates.

    :param atoms: Atomic label and cartesian coordinates in au.
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :type trans_mtx: Numpy Matrix
    :returns: tuple(<ψi | x | ψj>, <ψi | y | ψj>, <ψi | z | ψj> )
    """
    # x,y,z exponents value for the dipole
    exponents = [{'e': 1, 'f': 0, 'g': 0}, {'e': 0, 'f': 1, 'g': 0},
                 {'e': 0, 'f': 0, 'g': 1}]

    dimCart = trans_mtx.shape[1]
    mtx_integrals_triang = tuple(calcMtxMultipoleP(atoms, cgfsN, rc, **kw)
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

if __name__ == "__main__":
    main()
