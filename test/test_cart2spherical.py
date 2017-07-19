# ===============================<>============================================
from itertools import groupby
from nac.basisSet.basisNormalization import compute_normalization_sphericals
from nac.common import (change_mol_units, triang2mtx)
from nac.integrals import (calcMtxOverlapP, calc_transf_matrix)
from nac.basisSet import create_dict_CGFs
from qmworks.parsers import readXYZ

import h5py
import numpy as np
# ===============================<>============================================
path_hdf5 = 'test/test_files/ethylene.hdf5'


def test_compare_with_cp2k():
    """
    Test overlap matrix transformation from cartesian to spherical
    """
    # Overlap matrix in cartesian coordinates
    basisname = "DZVP-MOLOPT-SR-GTH"
    # Molecular geometry in a.u.
    atoms = change_mol_units(readXYZ('test/test_files/ethylene.xyz'))
    dictCGFs = create_dict_CGFs(path_hdf5, basisname, atoms)

    # Compute the overlap matrix using the general multipole expression
    rs = calcMtxOverlapP(atoms, dictCGFs)
    mtx_overlap = triang2mtx(rs, 48)  # there are 48 Cartesian basis CGFs

    dict_global_norms = compute_normalization_sphericals(dictCGFs)

    with h5py.File(path_hdf5, 'r') as f5:
        transf_matrix = calc_transf_matrix(
            f5, atoms, basisname, dict_global_norms, 'cp2k')

    transpose = np.transpose(transf_matrix)

    test = np.dot(transf_matrix, np.dot(mtx_overlap, transpose))
    expected = np.load('test/test_files/overlap_ethylene_sphericals.npy')

    arr = test - expected
    print(np.diag(test))
    print(np.argmax(np.abs(arr)))
    print(test[27, 42])
    print(expected[27, 42])
    # print(arr[0])
    # print("With index i, j: ", n // 46, n % 46)
    # print("Val: ", val)
    # print([np.argmax(x) for x in arr])
    # print([np.max(x) for x in arr])

    assert np.allclose(test, expected)


if __name__ == '__main__':
    test_compare_with_cp2k()
