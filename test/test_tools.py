"""Test the workflows tools."""
import os
from pathlib import Path

import numpy as np
import pkg_resources as pkg
from qmflows.parsers import parse_string_xyz

from nac.workflows.tools import number_spherical_functions_per_atom

file_path = pkg.resource_filename('nac', '')
root = Path(os.path.split(file_path)[0])


def test_calc_sphericals():
    """Test the calculation of spherical functions."""
    with open(root / 'test/test_files/Cd33Se33.xyz', 'r') as f:
        mol = parse_string_xyz(f.read())
    path_hdf5 = root / "test/test_files/Cd33Se33.hdf5"
    xs = number_spherical_functions_per_atom(
        mol, "cp2k", "DZVP-MOLOPT-SR-GTH", path_hdf5)

    expected = np.concatenate((np.repeat(25, 33), np.repeat(13, 33)))

    assert np.array_equal(xs, expected)
