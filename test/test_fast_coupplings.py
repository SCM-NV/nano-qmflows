
from nac import calculateCoupling3Points
from nac.common import (femtosec2au, retrieve_hdf5_data)
from nac.schedule import compute_phases
from nac.integrals import correct_phases
from os.path import join

import numpy as np

path_hdf5 = 'test/test_files/ethylene.hdf5'
path_hdf5_test = 'test/test_files/test.hdf5'
path_traj_xyz = 'test/test_files/threePoints.xyz'


def create_paths_mos(project_name, i):
    """
    create the path to the nodes containing the MOs inside the HDF5.
    """
    root = join(project_name, 'point_{}'.format(i), 'cp2k', 'mo')

    return [join(root, 'eigenvalues'), join(root, 'coefficients')]


def test_obaraSaika():
    """
    Test the Obara-Saika scheme to compute overlap integrals and
    then cumputed the derivated coupling using them.
    """
    project_name = 'ethylene'

    # Read the Overlap matrices from the HDF5
    names_matrices = ['mtx_sji_t0', 'mtx_sij_t0']
    concat_paths = [join(project_name, 'overlaps_{}'.format(i), mtx)
                    for i in range(2) for mtx in names_matrices]

    overlaps = np.stack([retrieve_hdf5_data(path_hdf5, ps)
                         for ps in concat_paths])

    # Size of the overlaps matrices
    dim = 12

    # Compute all the phases
    mtx_phases = compute_phases(overlaps, 1, dim)

    # Correct the phases
    tensor = overlaps[0: 4]
    fixed_phase_overlaps = correct_phases(tensor, mtx_phases[0: 3])

    dt_au = femtosec2au
    rs = calculateCoupling3Points(dt_au, *fixed_phase_overlaps)

    expected = retrieve_hdf5_data(path_hdf5, join(project_name, 'coupling_0'))

    assert np.allclose(rs, expected)

if __name__ == "__main__":
    test_obaraSaika()
