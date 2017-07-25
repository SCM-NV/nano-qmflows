# ===============================<>============================================
from nac.basisSet.basisNormalization import compute_normalization_sphericals
from nac.common import (change_mol_units, triang2mtx)
from nac.integrals import (calcMtxOverlapP, calc_transf_matrix)
from nac.basisSet import create_dict_CGFs
from qmworks.parsers import readXYZ
from scipy import sparse

import h5py
import numpy as np
# ===============================<>============================================
path_hdf5 = 'test/test_files/Cd33Se33.hdf5'


def test_compare_with_cp2k():
    """
    Test overlap matrix transformation from cartesian to spherical
    """
    # Overlap matrix in cartesian coordinates
    basisname = "DZVP-MOLOPT-SR-GTH"
    # Molecular geometry in a.u.
    atoms = change_mol_units(readXYZ('test/test_files/Cd33Se33_fivePoints.xyz'))

    dictCGFs = create_dict_CGFs(path_hdf5, basisname, atoms)

    # Compute the overlap matrix using the general multipole expression
    rs = calcMtxOverlapP(atoms, dictCGFs)
    mtx_overlap = triang2mtx(rs, 1452)  # there are 1452 Cartesian basis CGFs

    dict_global_norms = compute_normalization_sphericals(dictCGFs)

    with h5py.File(path_hdf5, 'r') as f5:
        transf_mtx = calc_transf_matrix(
            f5, atoms, basisname, dict_global_norms, 'cp2k')

    # Use a sparse representation of the transformation matrix
    transf_mtx = sparse.csr_matrix(transf_mtx)
    transpose = transf_mtx.transpose()

    # Compare the results with CP2K overlap matrix
    test = transf_mtx.dot(sparse.csr_matrix.dot(mtx_overlap, transpose))
    expected = np.load('test/test_files/overlap_Cd33Se33_cp2k.npy')

    assert np.allclose(test, expected, atol=1e-5)

if __name__ == '__main__':
    test_compare_with_cp2k()
