
from nac.common import (binomial, fac, odd, product,
                        retrieve_hdf5_data, triang2mtx)
from os.path import join

import numpy as np

# ====================================<>=======================================
path_hdf5 = 'test/test_files/ethylene.hdf5'


def test_product():
    """ test product function """
    assert product(range(1, 6)) == 120


def test_odd():
    """ test odd function """
    assert all(map(odd, [-3, -1, 3, 7, 11, 23]))


def test_fac():
    """Test factorial function """
    assert all([fac(5) == 120, fac(3) == 6])


def test_binomial():
    """Test binomial function """
    assert all([binomial(7, 4) == 35, binomial(8, 4) == 70,
                binomial(8, 3) == 56])


def test_triang_to_dim2():
    """
    Test the expantion from a flatten upper triangular matrix to a complete
    2-dimensional matrix.
    """
    xs = [[0, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]]
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected = np.array(xs)
    mtx = triang2mtx(arr, 4)

    assert np.sum(expected - mtx) < 1e-8


def test_retrieve_hdf5():
    """
    Test the function to read data from the HDF5 file.
    """
    root = 'ethylene/point_3/cp2k/mo'
    ps = ['coefficients', 'eigenvalues']
    properties = [join(root, p) for p in ps]
    retrieve_hdf5_data(path_hdf5, properties)
    try:
        false_path = 'Nonexisting/node/path'
        retrieve_hdf5_data(path_hdf5, false_path)
    except KeyError:
        assert True
