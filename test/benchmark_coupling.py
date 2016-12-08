from nac.common import (change_mol_units, femtosec2au, retrieve_hdf5_data)
from nac.integrals import calculateCoupling3Points
from nac.schedule.components import (create_dict_CGFs, split_file_geometries)
from qmworks.parsers import parse_string_xyz

import h5py
import numpy as np
# ===============================<>============================================
path_hdf5 = 'test/test_files/ethylene.hdf5'
path_xyz = 'test/test_files/threePoints.xyz'


def benchmark_coupling():
    """
    run a single point calculation using CP2K and store the MOs.
    """
    str_geometries = split_file_geometries(path_xyz)
    geometries_ang = tuple(map(parse_string_xyz, str_geometries))
    geometries_au = tuple(map(change_mol_units, geometries_ang))
    dictCGFs = create_dict_CGFs(path_hdf5, "DZVP-MOLOPT-SR-GTH",
                                geometries_au[0])
    new_dictCGFs = {k: tuple(map(list, zip(*v)))
                    for k, v in dictCGFs.items()}

    trans_mtx = retrieve_hdf5_data(path_hdf5, 'ethylene/trans_mtx')
    dt_au = 1 * femtosec2au
    mo_paths = ['ethylene/point_{}/cp2k/mo/coefficients'.format(i)
                for i in range(3)]
    coefficients = retrieve_hdf5_data(path_hdf5, mo_paths)
    coefficients = [cs[:, 3: 43] for cs in coefficients]

    couplings = calculateCoupling3Points(geometries_au, coefficients,
                                         new_dictCGFs, dt_au, trans_mtx=trans_mtx)
    with h5py.File(path_hdf5) as f5:
        path_coupling = 'ethylene/coupling_0'
        couplings_expected = f5[path_coupling].value

    assert np.allclose(couplings, couplings_expected)


if __name__ == "__main__":
    benchmark_coupling()
