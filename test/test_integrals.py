#!/usr/bin/env python

from compute_integrals import compute_integrals_couplings, compute_integrals_multipole
import numpy as np
import os
import pkg_resources as pkg

# Environment data
file_path = pkg.resource_filename('nac', '')
root = os.path.split(file_path)[0]

path_xyz = os.path.join(root, "test_files/ethylene.xyz")
path_hdf5 = os.path.join(root, "test_files/ethylene.hdf5")
basis_name = "DZVP-MOLOPT-SR-GTH"


def test_overlap_multipole():
    ys = compute_integrals_multipole(path_xyz, path_hdf5, basis_name, "overlap")
    xs = compute_integrals_couplings(path_xyz, path_xyz, path_hdf5, basis_name)

    print("allclose: ", np.allclose(xs, ys))
