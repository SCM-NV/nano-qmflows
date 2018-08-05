from nac.common import (
    change_mol_units, retrieve_hdf5_data, triang2mtx)
from nac.integrals.overlapIntegral import calcMtxOverlapP
from nac.workflows.workflow_AbsortionSpectrum import (
    transform2Spherical, workflow_oscillator_strength)
from nac.workflows.initialization import initialize
from os.path import join
from qmflows.parsers import parse_string_xyz
from qmflows.utils import dict2Setting

import h5py
import numpy as np
import pytest
import os
import shutil


cp2k_main = dict2Setting({
    'cell_parameters': 28.0, 'potential': 'GTH-PBE',
    'basis': 'DZVP-MOLOPT-SR-GTH', 'specific':
    {'cp2k': {'force_eval':
              {'subsys': {'cell': {'periodic': 'None'}}, 'dft':
               {'print': {'mo': {'mo_index_range': '248 327'}},
                'scf': {'eps_scf': 0.0005, 'max_scf': 200,
                        'added_mos': 30}}}}},
    'cell_angles': [90.0, 90.0, 90.0]})

cp2k_guess = dict2Setting({
    'cell_parameters': 28.0, 'potential': 'GTH-PBE',
    'basis': 'DZVP-MOLOPT-SR-GTH', 'specific':
    {'cp2k': {'force_eval':
              {'subsys': {'cell': {'periodic': 'None'}},
               'dft': {'scf': {'eps_scf': 1e-06, 'ot':
                               {'minimizer': 'DIIS',
                                'n_diis': 7, 'preconditioner':
                                'FULL_SINGLE_INVERSE'},
                               'scf_guess': 'restart',
                               'added_mos': 0}}}}},
    'cell_angles': [90.0, 90.0, 90.0]})

# Environment data
basisname = 'DZVP-MOLOPT-SR-GTH'
path_traj_xyz = 'test/test_files/Cd33Se33_fivePoints.xyz'
scratch_path = 'scratch'
path_original_hdf5 = 'test/test_files/Cd33Se33.hdf5'
path_test_hdf5 = join(scratch_path, 'test.hdf5')
project_name = 'Cd33Se33'


@pytest.mark.slow
def test_couplings_and_oscillators():
    """
    Test couplings and oscillator strength for Cd33Se33
    """
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)
    try:
        shutil.copy('test/test_files/BASIS_MOLOPT', scratch_path)
        shutil.copy('test/test_files/GTH_POTENTIALS', scratch_path)

        # Run the actual test
        copy_basis_and_orbitals(path_original_hdf5, path_test_hdf5,
                                project_name)
        data = calculate_oscillators()
        # # Check oscillator
        # fij = list(*chain(*data[0]))[5]
        # assert abs(fij - 0.130748) < 1e-6

        assert data[0] == 1254

    finally:
        # remove tmp data and clean global config
        # shutil.rmtree(scratch_path)
        pass


def calculate_oscillators():
    """
    Compute a couple of couplings with the Levine algorithm
    using precalculated MOs.
    """
    initial_config = initialize(
        project_name, path_traj_xyz,
        basisname=basisname, path_basis=None,
        path_potential=None, enumerate_from=0,
        calculate_guesses='first', path_hdf5=path_test_hdf5,
        scratch_path=scratch_path)

    geometry_0 = initial_config['geometries'][0]
    mol = change_mol_units(parse_string_xyz(geometry_0))
    dictCGFs = initial_config['dictCGFs']
    trans_mtx = retrieve_hdf5_data(
        path_test_hdf5, initial_config['hdf5_trans_mtx'])
    dimCart = trans_mtx.shape[1]

    mtx_cart = triang2mtx(
        calcMtxOverlapP(mol, dictCGFs),
        dimCart)

    mtx_spheric = transform2Spherical(trans_mtx, mtx_cart)

    np.save('overlap_spheric.npy', mtx_spheric)

    return trans_mtx.shape[0]
    # data = workflow_oscillator_strength(
    #     'cp2k', project_name, cp2k_main, guess_args=cp2k_guess,
    #     nHOMO=50, couplings_range=(50, 30), initial_states=[50],
    #     energy_range=(0, 5),  # eV
    #     final_states=[[52]], **initial_config)

    # return data


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
    test_couplings_and_oscillators()
