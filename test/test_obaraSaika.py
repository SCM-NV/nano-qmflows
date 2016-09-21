
from nac import (calculateCoupling3Points, initialize)
from nac.common import (femtosec2au, retrieve_hdf5_data)
from os.path import join
from qmworks.parsers import parse_string_xyz

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
    basisname = "DZVP-MOLOPT-SR-GTH"
    basiscp2k = "test/test_files/BASIS_MOLOPT"
    potcp2k = "test/test_files/GTH_POTENTIALS"

    initial_config = initialize(project_name, path_traj_xyz,
                                basisname=basisname, path_basis=basiscp2k,
                                path_potential=potcp2k,
                                enumerate_from=0,
                                scratch='/tmp',
                                path_hdf5=path_hdf5_test,
                                calculate_guesses='first')

    string_geoms = initial_config['geometries']
    geometries = tuple(map(parse_string_xyz, string_geoms))

    dictCGFs = initial_config['dictCGFs']
    hdf5_trans_mtx = initial_config['hdf5_trans_mtx']
    trans_mtx = retrieve_hdf5_data(path_hdf5_test, hdf5_trans_mtx)
    trans_mtx = trans_mtx[3:43, :]
    dt_au = femtosec2au

    mo_paths = [create_paths_mos(project_name, i) for i in range(3)]
    mos = tuple(map(lambda j:
                    retrieve_hdf5_data(path_hdf5,
                                       mo_paths[j][1]), range(3)))

    mos = [xs[3:43, 3:43] for xs in mos]
    rs = calculateCoupling3Points(geometries, mos, dictCGFs, dt_au,
                                  trans_mtx)

    expected = retrieve_hdf5_data(path_hdf5, join(project_name, 'coupling_0'))

    assert np.sum(rs - expected) < 1e-7

if __name__ == "__main__":
    test_obaraSaika()
