/*
 * This module contains the implementation of several
 * kind of integrals used for non-adiabatic molecular dynamics,
 * including the overlaps integrals between different geometries
 * And the dipoles and quadrupoles to compute absorption spectra.
 * This module is based on libint and Eigen.
 * Copyright (C) 2018-2022 the Netherlands eScience Center.
 */

#ifndef INT_H_
#define INT_H_

#include "namd.hpp"

namd::Matrix compute_integrals_couplings(
  const std::string &path_xyz_1,
  const std::string &path_xyz_2,
  const std::string &path_hdf5,
  const std::string &basis_name
);

namd::Matrix compute_integrals_multipole(
  const std::string &path_xyz,
  const std::string &path_hdf5,
  const std::string &basis_name,
  const std::string &multipole
);

#endif // INT_H_
