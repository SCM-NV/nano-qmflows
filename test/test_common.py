
from nac.common import (binomial, fac, fromIndex, odd, product,
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


def test_triangular_from_index():
    """
    Test the convertion from a symmetric 2-dimensional matrix to a flattern
    upper triangular representation as shown in the following scheme:

    (0,0) 0  (0, 1) 1 (0, 2) 2 .... (0, n) n
             (1, 1) n+1.............(1, n) 2*n - 1
                                    (n, n) (n^2 + n) / 6
    """
    shape = (4, 4)
    fun = lambda x: fromIndex(x, shape)
    x0_3 = fun((0, 3)) == 3
    x1_1 = fun((1, 1)) == 4
    x2_2 = fun((2, 2)) == 7
    x2_3 = fun((2, 3)) == 8
    x3_3 = fun((3, 3)) == 9

    xs = [x0_3, x1_1, x2_2, x2_3, x3_3]
    assert all(xs)


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
