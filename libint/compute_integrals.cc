// This module contains the implementation of several
// kind of integrals used for non-adiabatic molecular dynamics,
// including the overlaps integrals between different geometries
// And the dipoles and quadrupoles to compute absorption spectra.

// This module is based on libint, Eigen and pybind11.

// Copyright (C) 2019 the Netherlands eScience Center.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// integrals library
#include <libint2.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

// Eigen matrix algebra library
#include <Eigen/Dense>

// HDF5 funcionality
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

// Constants
#include "namd.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;
using HighFive::Attribute;
using HighFive::DataSet;
using HighFive::File;
using libint2::Atom;
using libint2::BasisSet;
using libint2::Operator;
using libint2::Shell;
using namd::CP2K_Basis_Atom;
using namd::map_elements;
using namd::Matrix;
using std::string;

Matrix compute_integrals_couplings(const string &path_xyz_1,
                                   const string &path_xyz_2,
                                   const string &path_hdf5,
                                   const string &basis_name);

std::vector<Atom> read_xyz_from_file(const string &path_xyz) {
  // Read molecule in XYZ format from file
  std::ifstream input_file(path_xyz);
  return libint2::read_dotxyz(input_file);
}

int main() {

  string path_xyz = "../test/test_files/ethylene.xyz";
  string path_hdf5 = "../test/test_files/ethylene.hdf5";
  string basis_name = "DZVP-MOLOPT-SR-GTH";
  string dataset_name = "ethylene/point_n";

  auto xs =
      compute_integrals_couplings(path_xyz, path_xyz, path_hdf5, basis_name);
}

// OpenMP or multithread computations
namespace libint2 {
auto nthreads = 1;

/// fires off \c nthreads instances of lambda in parallel
template <typename Lambda> void parallel_do(Lambda &lambda) {
#ifdef _OPENMP
#pragma omp parallel
  {
    auto thread_id = omp_get_thread_num();
    lambda(thread_id);
  }
#else // use C++11 threads
  std::vector<std::thread> threads;
  for (auto thread_id = 0; thread_id != libint2::nthreads; ++thread_id) {
    if (thread_id != nthreads - 1)
      threads.push_back(std::thread(lambda, thread_id));
    else
      lambda(thread_id);
  } // threads_id
  for (auto thread_id = 0; thread_id < nthreads - 1; ++thread_id)
    threads[thread_id].join();
#endif
}
} // namespace libint2

void set_nthread() {

  using libint2::nthreads;
  nthreads = std::thread::hardware_concurrency();

#if defined(_OPENMP)
  omp_set_num_threads(nthreads);
#endif
  std::cout << "Will scale over " << nthreads
#if defined(_OPENMP)
            << " OpenMP"
#else
            << " C++11"
#endif
            << " threads" << std::endl;
}

size_t nbasis(const std::vector<libint2::Shell> &shells) {
  size_t n = 0;
  for (const auto &shell : shells)
    n += shell.size();
  return n;
}

size_t max_nprim(const std::vector<libint2::Shell> &shells) {
  size_t n = 0;
  for (auto shell : shells)
    n = std::max(shell.nprim(), n);
  return n;
}

int max_l(const std::vector<libint2::Shell> &shells) {
  int l = 0;
  for (auto shell : shells)
    for (auto c : shell.contr)
      l = std::max(c.l, l);
  return l;
}

std::vector<size_t>
map_shell_to_basis_function(const std::vector<Shell> &shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for (auto shell : shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}

Matrix compute_overlaps_for_couplings(const std::vector<Shell> &shells_1,
                                      const std::vector<Shell> &shells_2) {
  // Compute the overlap integrals between two set of shells at different
  // atomic positions

  // Distribute the computations among the available threads
  using libint2::nthreads;

  const auto n = nbasis(shells_1);
  Matrix result(n, n);

  // construct the overlap integrals engine
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] = libint2::Engine(Operator::overlap, max_nprim(shells_1),
                               max_l(shells_1), 0);

  for (auto i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  auto shell2bf = map_shell_to_basis_function(shells_1);

  // Function to compute the integrals in parallel
  auto compute = [&](int thread_id) {
    // buf[0] points to the target shell set after every call  to
    // engine.compute()
    const auto &buf = engines[thread_id].results();

    // loop over unique shell pairs, {s1,s2}
    for (auto s1 = 0; s1 != shells_1.size(); ++s1) {

      auto bf1 = shell2bf[s1]; // first basis function in this shell
      auto n1 = shells_1[s1].size();

      for (auto s2 = 0; s2 != shells_2.size(); ++s2) {
        auto acc = s2 + s1 * shells_1.size();
        if (acc % nthreads != thread_id)
          continue;

        // extract basis
        auto bf2 = shell2bf[s2];
        auto n2 = shells_2[s2].size();

        // compute shell pair and return pointer to the buffer
        engines[thread_id].compute(shells_1[s1], shells_2[s2]);

        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        result.block(bf1, bf2, n1, n2) = buf_mat;
      }
    }
  }; // compute lambda

  libint2::parallel_do(compute);

  return result;
}

template <Operator operator_type,
          typename OperatorParams = typename libint2::operator_traits<
              operator_type>::oper_params_type>
std::vector<Matrix> compute_multipoles(
    const std::vector<libint2::Shell> &shells,
    OperatorParams oparams =
        OperatorParams()) { // Compute different type of multipole integrals in
                            // different threads
  using libint2::nthreads;

  const unsigned int nopers = libint2::operator_traits<operator_type>::nopers;

  // number of shells
  const auto n = nbasis(shells);

  std::vector<Matrix> result(nopers);
  for (auto &r : result)
    r = Matrix::Zero(n, n);

  // construct the multipole engine integrals engine
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] =
      libint2::Engine(operator_type, max_nprim(shells), max_l(shells), 0);

  // pass operator params to the engines
  engines[0].set_params(oparams);
  for (auto i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  auto shell2bf = map_shell_to_basis_function(shells);

  // Function to compute the integrals in parallel
  auto compute = [&](int thread_id) {
    // buf[0] points to the target shell set after every call  to
    // engines.compute()
    const auto &buf = engines[thread_id].results();

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over
    // Hermitian operators: (1|2) = (2|1)
    for (auto s1 = 0; s1 != shells.size(); ++s1) {

      auto bf1 = shell2bf[s1]; // first basis function in this shell
      auto n1 = shells[s1].size();

      for (auto s2 = 0; s2 <= s1; ++s2) {
        // Select integrals for current thread
        auto acc = s2 + s1 * shells.size();
        if (acc % nthreads != thread_id)
          continue;

        auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        // compute shell pair
        engines[thread_id].compute(shells[s1], shells[s2]);

        for (auto op = 0; op != nopers; ++op) {
          // "map" buffer to a const Eigen Matrix, and copy it to the
          // corresponding blocks of the result
          Eigen::Map<const Matrix> buf_mat(buf[op], n1, n2);
          result[op].block(bf1, bf2, n1, n2) = buf_mat;
          if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding
                        // {s2,s1} block, note the transpose!
            result[op].block(bf2, bf1, n2, n1) = buf_mat.transpose();
        }
      }
    }
  }; // compute lambda
  libint2::parallel_do(compute);

  return result;
}

std::vector<int> read_basisFormat(const string &basisFormat) {
  // Transform the string containing the basis format for CP2K, into a vector of
  // strings
  std::vector<int> rs;
  for (auto x : basisFormat) {
    if (std::isdigit(x))
      // convert char to int
      rs.push_back(x - '0');
  }
  return rs;
}

CP2K_Basis_Atom read_basis_from_hdf5(const string &path_file,
                                     const string &symbol,
                                     const string &basis) {
  // Read a basis set from HDF5 as a matrix

  std::vector<std::vector<double>> coefficients;
  std::vector<double> exponents;
  string format;
  try {

    // Open an existing HDF5 File
    File file(path_file, File::ReadOnly);

    // build paths to the coefficients and exponents
    string root = "cp2k/basis/" + symbol + "/" + basis;
    string path_coefficients = root + "/coefficients";
    string path_exponents = root + "/exponents";

    // Get the dataset
    DataSet dataset_cs = file.getDataSet(path_coefficients);
    DataSet dataset_es = file.getDataSet(path_exponents);
    Attribute attr = dataset_cs.getAttribute("basisFormat");

    // extract data from the
    dataset_cs.read(coefficients);
    dataset_es.read(exponents);
    attr.read(format);

  } catch (HighFive::Exception &err) {
    // catch and print any HDF5 error
    std::cerr << err.what() << std::endl;
  }
  return CP2K_Basis_Atom{symbol, coefficients, exponents,
                         read_basisFormat(format)};
}

std::vector<string> get_unique_symbols(const std::vector<Atom> &atoms) {
  // Return a set of unique symbols
  std::vector<int> elements;
  std::transform(atoms.begin(), atoms.end(), std::back_inserter(elements),
                 [](const Atom &at) { return at.atomic_number; });
  // Unique set of elements
  std::unordered_set<int> set(elements.begin(), elements.end());

  // create a unique vector of symbols
  std::vector<string> symbols;
  std::transform(set.cbegin(), set.cend(), std::back_inserter(symbols),
                 [](int x) { return map_elements[x]; });
  return symbols;
}

std::unordered_map<string, CP2K_Basis_Atom>
create_map_symbols_basis(const string &path_hdf5,
                         const std::vector<Atom> &atoms, const string &basis) {
  // Function to generate a map from symbols to basis specification

  std::unordered_map<string, CP2K_Basis_Atom> dict;

  // Select the unique atomic symbols
  std::vector<string> symbols = get_unique_symbols(atoms);
  for (const auto &at : symbols)
    dict[at] = read_basis_from_hdf5(path_hdf5, at, basis);

  return dict;
}

std::vector<Shell> create_shells_for_atom(const CP2K_Basis_Atom &data,
                                          const Atom &atom) {
  // Create the shell specification for a given atom.
  // The CP2K basis format is defined by a vector of integers, for each atom.
  // For example For the C atom and the Basis DZVP-MOLOPT-GTH the basis format
  // is:
  //  2 0 2 7 2 2 1
  // where:
  //   * 2 is the Principal quantum number
  //   * 0 is the minimum angular momemtum l
  //   * 2 is the maximum angular momentum l
  //   * 7 is the number of total exponents
  //   * 2 Contractions of S Gaussian Orbitals
  //   * 2 Contractions of P Gaussian Orbitals
  //   * 1 Contraction of D Gaussian Orbital
  // Note: From element 4 onwards are define the number of contracted for each
  // quantum number.
  std::vector<int> basis_format = data.basis_format;
  std::vector<Shell> shells;

  auto acc = 0;
  for (auto i = 0; i + 4 < basis_format.size(); i++) {
    for (auto j = 0; j < basis_format[i + 4]; j++) {
      shells.push_back({data.exponents,
                        {// compute integrals in sphericals
                         {i, true, data.coefficients[acc]}},
                        // Atomic Coordinates
                        {{atom.x, atom.y, atom.z}}});
      acc += 1;
    }
  }
  return shells;
}

std::vector<Shell> make_cp2k_basis(const std::vector<Atom> &atoms,
                                   const string &path_hdf5,
                                   const string &basis) {
  // Make the shell for a CP2K specific basis

  std::vector<Shell> shells;

  // set of symbols
  std::vector<string> symbols = get_unique_symbols(atoms);

  // Read basis set data from the HDF5
  std::unordered_map<string, CP2K_Basis_Atom> dict =
      create_map_symbols_basis(path_hdf5, atoms, basis);

  for (const auto &atom : atoms) {

    CP2K_Basis_Atom data = dict[map_elements[atom.atomic_number]];
    auto xs = create_shells_for_atom(data, atom);
    shells.insert(shells.end(), xs.begin(), xs.end());
  }

  return shells;
}

Matrix compute_integrals_couplings(const string &path_xyz_1,
                                   const string &path_xyz_2,
                                   const string &path_hdf5,
                                   const string &basis_name) {
  // Compute the overlap integrals for the molecule define in `path_xyz` using
  // the `basis_name`

  set_nthread();
  std::vector<Atom> mol_1 = read_xyz_from_file(path_xyz_1);
  std::vector<Atom> mol_2 = read_xyz_from_file(path_xyz_2);

  auto shells_1 = make_cp2k_basis(mol_1, path_hdf5, basis_name);
  auto shells_2 = make_cp2k_basis(mol_2, path_hdf5, basis_name);

  // safe to use libint now
  libint2::initialize();

  // compute Overlap integrals
  auto S = compute_overlaps_for_couplings(shells_1, shells_2);

  // stop using libint2
  libint2::finalize();

  return S;
}

std::array<double, 3> calculate_center_of_mass(const std::vector<Atom> &atoms) {
  // Compute the center of mass for atoms
  auto n = atoms.size();

  std::array<double, 3> rs{0, 0, 0};
  for (auto i = 0; i < n; i++) {
    auto at = atoms[i];
    auto z = double(at.atomic_number);
    rs[0] += at.x * z;
    rs[1] += at.y * z, rs[2] += at.z * z;
  }

  auto m =
      std::accumulate(atoms.begin(), atoms.end(), 0, [](double acc, Atom at) {
        return acc + double(at.atomic_number);
      });
  return {rs[0] / m, rs[1] / m, rs[2] / m};
}

std::vector<Matrix> select_multipole(const std::vector<Atom> &atoms,
                                     const std::vector<Shell> &shells,
                                     const string &multipole) {
  // Compute multipole at the center of mass
  std::array<double, 3> center = calculate_center_of_mass(atoms);

  if (multipole == "overlap")
    return compute_multipoles<Operator::overlap>(shells);
  else if (multipole == "dipole")
    return compute_multipoles<Operator::emultipole1>(shells, center);
  else if (multipole == "quadrupole")
    return compute_multipoles<Operator::emultipole2>(shells, center);
}

Matrix compute_integrals_multipole(const string &path_xyz,
                                   const string &path_hdf5,
                                   const string &basis_name,
                                   const string &multipole) {
  // Compute the overlap integrals for the molecule define in `path_xyz` using
  // the `basis_name`

  set_nthread();
  std::vector<Atom> mol = read_xyz_from_file(path_xyz);

  auto shells = make_cp2k_basis(mol, path_hdf5, basis_name);

  // safe to use libint now
  libint2::initialize();

  // compute Overlap integrals
  auto matrices = select_multipole(mol, shells, multipole);

  // stop using libint2
  libint2::finalize();

  Matrix super_matrix(matrices[0].rows() * matrices.size(), matrices[0].cols());
  for (auto op = 0; op != matrices.size(); ++op) {
    auto i = op * matrices[0].rows();
    super_matrix.block(i, 0, matrices[op].rows(), matrices[op].cols()) =
        matrices[op];
  }

  return super_matrix;
}

PYBIND11_MODULE(compute_integrals, m) {
  m.doc() = "Compute integrals using libint2 see: "
            "https://github.com/evaleev/libint/wiki";

  m.def("compute_integrals_couplings", &compute_integrals_couplings,
        py::return_value_policy::reference_internal);

  m.def("compute_integrals_multipole", &compute_integrals_multipole,
        py::return_value_policy::reference_internal);
}
