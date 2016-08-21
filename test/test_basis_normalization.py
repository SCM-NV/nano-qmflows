
# ===============================<>============================================
from nac.basisSet import createNormalizedCGFs
from nac.common import AtomXYZ
from nac.integrals.overlapIntegral import sijContracted
from multiprocessing import Pool
import h5py
import numpy as np
# ===============================<>============================================
path_hdf5 = 'test/test_files/ethylene.hdf5'


def test_normalization():
    """
    A CGFs is given by
    |fi> = sum ci* ni* x^lx * y^ly * z ^lz * exp(-ai * R^2)
    After normalization <fi|fi> = 1
    """
    basis_name = "DZVP-MOLOPT-SR-GTH"
    package_name = "cp2k"
    mol = [AtomXYZ(symbol='cd', xyz=[])]

    with h5py.File(path_hdf5) as f5:
        dict_cgfs = createNormalizedCGFs(f5, basis_name, package_name, mol)

    xyz = [0] * 3
    # iterate over the primitive gaussian functions
    tuples = [(xyz, g) for g in dict_cgfs['cd']]

    with Pool() as p:
        sii = p.starmap(sijContracted, [(t, t) for t in tuples])

    # Expected a vector size which size is the number of CGFs
    # with the integrals
    # [<cgfs_0| cgfs_0> , <cgfs_1 | cgfs_1>...] which is
    # [1,1,...]
    expected = np.ones(len(tuples))

    assert np.sum(expected - sii) < 1.0e-8
