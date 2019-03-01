#!/usr/bin/env python

from compute_integrals import compute_integrals_couplings, compute_integrals_multipole
import numpy as np

path_xyz = "test_files/ethylene.xyz"

path_hdf5 = "test_files/ethylene.hdf5"

basis_name = "DZVP-MOLOPT-SR-GTH"

ys = compute_integrals_multipole(path_xyz, path_hdf5, basis_name, "overlap")
xs = compute_integrals_couplings(path_xyz, path_xyz, path_hdf5, basis_name)

print(xs[0, :])
print(ys[0, :])
print("allclose: ", np.allclose(xs, ys))
