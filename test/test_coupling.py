
from nac.basisSet.basisNormalization import createNormalizedCGFs
from nac.common import change_mol_units
from nac.schedule.components import split_file_geometries
from nac.schedule.scheduleCoupling import lazy_schedule_couplings
from pymonad import curry
from qmworks import run
from qmworks.parsers import parse_string_xyz
from utilsTest import try_to_remove

import h5py
import numpy as np
# ===============================<>============================================
path_hdf5 = 'test/test_files/ethylene.hdf5'
path_hdf5_test = 'test/test_files/test.hdf5'
path_xyz = 'test/test_files/threePoints.xyz'


def is_antisymmetric(arr):
    """
    Check if a matrix is antisymmetric. Notice that the coupling matrix has
    all the diagonal elements equal to zero.
    """
    return np.sum(arr + np.transpose(arr)) < 1.0e-8


def test_lazy_coupling():
    """
    The matrix containing the derivative coupling must be antisymmetric and
    it should be equal to the already known value stored in the HDF5 file:
    `test/test_files/ethylene.hdf5`
    """
    basis_name = "DZVP-MOLOPT-SR-GTH"
    package_name = "cp2k"
    parser = curry(parse_string_xyz)
    # function composition
    new_fun = change_mol_units * parser
    geometries = tuple(map(new_fun, split_file_geometries(path_xyz)))

    # The MOs were already computed an stored in ethylene.hdf5
    root_css = 'ethylene/point_{}/cp2k/mo/coefficients'.format
    root_es = 'ethylene/point_{}/cp2k/mo/exponents'.format
    mo_paths = [[root_es(i), root_css(i)] for i in range(3)]

    with h5py.File(path_hdf5) as f5, h5py.File(path_hdf5_test, 'w') as f6:
        dict_cgfs = createNormalizedCGFs(f5, basis_name, package_name,
                                         geometries[0])
        # Copy the MOs and trans_mtx to the temporal hdf5
        f6.create_group('ethylene')
        f5.copy('ethylene/trans_mtx', f6['ethylene'])
        for k in range(3):
            p = 'ethylene/point_{}'.format(k)
            f5.copy(p, f6['ethylene'])

    output_folder = 'ethylene'
    rs = lazy_schedule_couplings(0, path_hdf5_test, dict_cgfs,
                                 geometries, mo_paths, dt=1,
                                 hdf5_trans_mtx='ethylene/trans_mtx',
                                 output_folder=output_folder,
                                 nCouplings=40)
    path_coupling = run(rs)

    with h5py.File(path_hdf5, 'r') as f5, h5py.File(path_hdf5_test, 'r') as f6:
        expected = f5[path_coupling].value
        arr = f6[path_coupling].value

    # remove the hdf5 file used for testing
    try_to_remove(path_hdf5_test)

    assert is_antisymmetric(arr) and (np.sum(arr - expected) < 1.0e-8)
