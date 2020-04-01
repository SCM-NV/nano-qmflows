"""Test the workflows tools."""
import numpy as np
from qmflows.parsers import parse_string_xyz

from nanoqm.common import number_spherical_functions_per_atom

from .utilsTest import PATH_TEST


def test_calc_sphericals():
    """Test the calculation of spherical functions."""
    with open(PATH_TEST / 'Cd33Se33.xyz', 'r') as f:
        mol = parse_string_xyz(f.read())
    path_hdf5 = PATH_TEST / "Cd33Se33.hdf5"
    xs = number_spherical_functions_per_atom(
        mol, "cp2k", "DZVP-MOLOPT-SR-GTH", path_hdf5)

    expected = np.concatenate((np.repeat(25, 33), np.repeat(13, 33)))

    assert np.array_equal(xs, expected)
