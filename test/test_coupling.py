from nac import (calculate_mos, initialize)
from nac.workflows.workflow_coupling import generate_pyxaid_hamiltonians
from nose.plugins.attrib import attr
from os.path import join
from qmworks import (run, Settings)
from utilsTest import try_to_remove

import h5py
import numpy as np
import os
import shutil

# ===============================<>============================================
path_hdf5 = 'test/test_files/ethylene.hdf5'
path_hdf5_test = 'test/test_files/test.hdf5'
path_xyz = 'test/test_files/threePoints.xyz'
project_name = 'ethylene'


def is_antisymmetric(arr):
    """
    Check if a matrix is antisymmetric. Notice that the coupling matrix has
    all the diagonal elements equal to zero.
    """
    return np.sum(arr + np.transpose(arr)) < 1.0e-8


@attr('slow')
def test_workflow_coupling():
    """
    run a single point calculation using CP2K and store the MOs.
    """
    home = os.path.expanduser('~')  # HOME Path
    scratch_path = join(home, '.test_qmworks')
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)
    try:
        shutil.copy('test/test_files/BASIS_MOLOPT', scratch_path)
        shutil.copy('test/test_files/GTH_POTENTIALS', scratch_path)
        basiscp2k = join(scratch_path, 'BASIS_MOLOPT')
        potcp2k = join(scratch_path, 'GTH_POTENTIALS')

        initial_config = initialize(project_name, path_xyz,
                                    "DZVP-MOLOPT-SR-GTH",
                                    path_basis=basiscp2k,
                                    path_potential=potcp2k,
                                    calculate_guesses=None,
                                    path_hdf5=path_hdf5_test,
                                    scratch=scratch_path)

        # create Settings for the Cp2K Jobs
        cp2k_args = Settings()
        cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
        cp2k_args.potential = "GTH-PBE"
        cp2k_args.cell_parameters = [12.74] * 3
        dft = cp2k_args.specific.cp2k.force_eval.dft
        dft.scf.added_mos = 40
        dft["print"]["mo"]["mo_index_range"] = "7 46"
        dft.scf.diagonalization.jacobi_threshold = 1e-6

        force = cp2k_args.specific.cp2k.force_eval
        basis_pot = initial_config['package_config']
        force.dft.basis_set_file_name = basis_pot['basis']
        force.dft.potential_file_name = basis_pot['potential']

        # call tests
        fun_calculte_mos(cp2k_args, initial_config)
        fun_workflow_coupling(cp2k_args, initial_config)
    finally:
        # remove tmp data and clean global config
        shutil.rmtree(scratch_path)


def fun_calculte_mos(cp2k_args, ds):
    """
    """
    paths = calculate_mos('cp2k', ds['geometries'], project_name,
                          path_hdf5_test, ds['traj_folders'], cp2k_args,
                          package_config=ds['package_config'])
    run(paths)
    with h5py.File(path_hdf5) as f5, h5py.File(path_hdf5_test) as f6:
        path_css = 'ethylene/point_1/cp2k/mo/coefficients'
        path_es = 'ethylene/point_1/cp2k/mo/eigenvalues'
        es_expected = f5[path_es].value
        es_test = f6[path_es].value
        css_expected = f5[path_css].value
        css_test = f6[path_css].value

    # There are 46 Orbitals Stored in ethylene.hdf5 file
    delta_css = abs(np.sum(css_expected[:, 6:] - css_test))
    delta_es = abs(np.sum(es_expected[6:] - es_test))

    assert delta_es < 1e-6 and delta_css < 1e-6


@try_to_remove([path_hdf5_test])
def fun_workflow_coupling(cp2k_args, initial_config):
    """
    Call cp2k and calculated the coupling for an small molecule.
    """
    generate_pyxaid_hamiltonians('cp2k', project_name,
                                 cp2k_args, nCouplings=40,
                                 **initial_config)

    with h5py.File(path_hdf5) as f5, h5py.File(path_hdf5_test) as f6:
        path_coupling = 'ethylene/coupling_0'
        coupling_expected = f5[path_coupling].value
        coupling_test = f6[path_coupling].value

        tolerance = 1e-6
        assert ((abs(np.sum(coupling_expected - coupling_test)) < tolerance)
                and is_antisymmetric(coupling_test))


if __name__ == "__main__":
    test_workflow_coupling()
