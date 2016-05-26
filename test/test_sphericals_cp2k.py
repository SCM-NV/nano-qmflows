
from qmworks.hdf5.quantumHDF5 import cp2k2hdf5
from qmworks.parsers.xyzParser import readXYZ

from os.path import join
import h5py
import numpy as np
import os

# ========================<>=======================
from nac.common import InputKey
from nac.integrals.overlapIntegral import calcMtxOverlapP
from nac.integrals.spherical_Cartesian_cgf import calc_transf_matrix

from utilsTest import  (change_mol_units, create_dict_CGFs, triang2mtx, try_to_remove)
# ========================<>=======================

path_hdf5 = 'test_files/test.hdf5'


def save_overlap(pathOverlap, f5, path, nOrbitals):
    key = InputKey('overlap', [pathOverlap, nOrbitals, path])
    try_to_remove(path_hdf5)
    cp2k2hdf5(f5, [key])


def test_load_overlap():
    """
    Read the overlap matrix in Spherical coordinates from Cp2k output
    """
    path = join('cp2k', 'test', 'ethylene', 'overlap')
    with h5py.File(path_hdf5, chunks=True) as f5:
        try:
            save_overlap("test_files/overlap_ethylene_cp2k.out", f5, path, 46)
            f5[path]
            assert True
        except RuntimeError:
            assert False


def test_sphericals_ethylene():
    """
    The Overlap in spherical is equal to Cs S C^(+)
    Where Cs are the coefficients in spherical coordinates
    """
    basis = 'DZVP-MOLOPT-SR-GTH'
    mol = readXYZ('test_files/ethylene_au.xyz')
    labels = [at.symbol for at in mol]
    pathBasis = os.environ['BASISCP2K']
    path = join('cp2k', 'test', 'ethylene', 'overlap')

    try_to_remove(path_hdf5)
    with h5py.File(path_hdf5, chunks=True) as f5:
        dictCGFs = create_dict_CGFs(f5, cp2k2hdf5,
                                    pathBasis, basis, 'cp2k', mol)
        css = calc_transf_matrix(f5, mol, basis, 'cp2k')

        pathOverlap = "test_files/overlap_ethylene_cp2k.out"
        save_overlap(pathOverlap, f5, path, 46)
        dset = f5[path]
        overlapCp2k = dset[...]

    cgfsN = [dictCGFs[l] for l in labels]
    dim = sum(len(xs) for xs in cgfsN)
    mtxP = triang2mtx(calcMtxOverlapP(mol, cgfsN), dim)
    css_t = np.transpose(css)

    overlapS = np.dot(css, np.dot(mtxP, css_t))

    try_to_remove(path_hdf5)

    err = overlapS - overlapCp2k
    norm = np.linalg.norm(err)
    print("The err norm is: \n", norm)
    assert  norm < 1.0e-5


def test_sphericals_Cd16Se13_6HCOO():
    """
    The Overlap in spherical is equal to Cs S C^(+)
    Where Cs are the coefficients in spherical coordinates
    """
    basis = 'DZVP-MOLOPT-SR-GTH'
    # nOrbitals = 46
    mol = readXYZ('test_files/Cd16Se13_6HCOO.xyz')
    mol = change_mol_units(mol)  # Angstrom to a.u.
    labels = [at.symbol for at in mol]
    pathBasis = os.environ['BASISCP2K']
    path = join('cp2k', 'test', 'Cd16Se13_6HCOO', 'overlap')

    try_to_remove(path_hdf5)
    with h5py.File(path_hdf5, chunks=True) as f5:
        dictCGFs = create_dict_CGFs(f5, cp2k2hdf5,
                                    pathBasis, basis, 'cp2k', mol)
        css = calc_transf_matrix(f5, mol, basis, 'cp2k')

        pathOverlap = "test_files/overlap_Cd16Se13_6HCOO_cp2k.out"
        save_overlap(pathOverlap, f5, path, 833)
        dset = f5[path]
        overlapCp2k = dset[...]

    cgfsN = [dictCGFs[l] for l in labels]
    dim = sum(len(xs) for xs in cgfsN)
    mtxP = triang2mtx(calcMtxOverlapP(mol, cgfsN), dim)
    css_t = np.transpose(css)

    overlapS = np.dot(css, np.dot(mtxP, css_t))

    try_to_remove(path_hdf5)

    err = overlapS - overlapCp2k
    norm = np.linalg.norm(err)
    print("The err norm is: \n", norm)
    assert  norm < 1.0e-3
