/*
 * This module contains the implementation of several
 * kind of integrals used for non-adiabatic molecular dynamics,
 * including the overlaps integrals between different geometries
 * And the dipoles and quadrupoles to compute absorption spectra.
 * This module is based on libint and Eigen.
 * Copyright (C) 2018-2022 the Netherlands eScience Center.
 */

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x03070000
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "namd.hpp"

using HighFive::Attribute;
using HighFive::DataSet;
using HighFive::File;
using HighFive::Group;
using libint2::Atom;
using libint2::BasisSet;
using libint2::Operator;
using libint2::Shell;
using namd::CP2K_Contractions;
using namd::CP2K_Basis_Atom;
using namd::map_elements;
using namd::valence_electrons;
using namd::Matrix;
using std::string;

std::vector<Atom> read_xyz_from_file(const string &path_xyz) {
  // Read molecule in XYZ format from file
  std::ifstream input_file(path_xyz);
  return libint2::read_dotxyz(input_file);
}

/**
 * \brief Compute the coupling integrals for the 2 given molecular geometries
 * and basis_name.
 */
Matrix compute_integrals_couplings(const string &path_xyz_1,
                                   const string &path_xyz_2,
                                   const string &path_hdf5,
                                   const string &basis_name);

int main() {
  const string path_xyz = "../test/test_files/ethylene.xyz";
  const string path_hdf5 = "../test/test_files/ethylene.hdf5";
  const string basis_name = "DZVP-MOLOPT-SR-GTH";
  const string dataset_name = "ethylene/point_n";
  auto xs = compute_integrals_couplings(path_xyz, path_xyz, path_hdf5, basis_name);
}

// OpenMP or multithread computations
namespace libint2 {
int nthreads = 1;

/**
 * \brief fires off \c nthreads instances of lambda in parallel
 */
template <typename Lambda> void parallel_do(Lambda &lambda) {
#ifdef _OPENMP
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    lambda(thread_id);
  }
#else // use C++11 threads
  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id != libint2::nthreads; ++thread_id) {
    if (thread_id != nthreads - 1) {
      threads.push_back(std::thread(lambda, thread_id));
    }
    else {
      lambda(thread_id);
    }
  } // threads_id
  for (int thread_id = 0; thread_id < nthreads - 1; ++thread_id) {
    threads[thread_id].join();
  }
#endif
}
} // namespace libint2

/**
 * \brief Set the number of thread to use
 */
void set_nthread() {
#if defined(_OPENMP)
  using libint2::nthreads;
  nthreads = std::thread::hardware_concurrency();
  omp_set_num_threads(nthreads);
#endif
}

/**
 * \brief Get the number of basis
 */
size_t nbasis(const std::vector<libint2::Shell> &shells) {
  return std::accumulate(
    shells.cbegin(),
    shells.cend(),
    0,
    [](const size_t acc, const libint2::Shell &shell) { return acc + shell.size(); });
}

/**
 * \brief compute the maximum number of n-primitives
 */
size_t max_nprim(const std::vector<libint2::Shell> &shells) {
  size_t n = 0;
  for (const auto &shell : shells) {
    n = std::max(shell.nprim(), n);
  }
  return n;
}

/**
 * \brief compute the maximum number of l-primitives
 */
int max_l(const std::vector<libint2::Shell> &shells) {
  int l = 0;
  for (const auto &shell : shells) {
    for (const auto &c : shell.contr) {
      l = std::max(c.l, l);
    }
  }
  return l;
}

/**
 * \brief Count the number of basis functions per shell.
 */
std::vector<size_t>
map_shell_to_basis_function(const std::vector<Shell> &shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  auto n = 0;
  for (const auto &shell : shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}
/**
 * \brief Compute the overlap integrals between two set of shells at different
 * atomic positions
 */
Matrix compute_overlaps_for_couplings(const std::vector<Shell> &shells_1,
                                      const std::vector<Shell> &shells_2) {
  // Distribute the computations among the available threads
  using libint2::nthreads;

  const auto n = nbasis(shells_1);
  Matrix result(n, n);

  // construct the overlap integrals engine
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] = libint2::Engine(Operator::overlap, max_nprim(shells_1),
                               max_l(shells_1), 0);

  for (int i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  const auto shell2bf = map_shell_to_basis_function(shells_1);

  // Function to compute the integrals in parallel
  auto compute = [&](const int thread_id) {
    // buf[0] points to the target shell set after every call  to
    // engine.compute()
    const auto &buf = engines[thread_id].results();

    // loop over unique shell pairs, {s1,s2}
    for (int s1 = 0; s1 != static_cast<int>(shells_1.size()); ++s1) {

      int bf1 = shell2bf[s1]; // first basis function in this shell
      int n1 = shells_1[s1].size();

      for (int s2 = 0; s2 != static_cast<int>(shells_2.size()); ++s2) {
        int acc = s2 + s1 * shells_1.size();
        if (acc % nthreads != thread_id)
          continue;

        // extract basis
        int bf2 = shell2bf[s2];
        int n2 = shells_2[s2].size();

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

  constexpr unsigned int nopers =
      libint2::operator_traits<operator_type>::nopers;

  // number of shells
  const auto n = nbasis(shells);

  std::vector<Matrix> result(nopers);
  for (auto &r : result) {
    r = Matrix::Zero(n, n);
  }

  // construct the multipole engine integrals engine
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] = libint2::Engine(operator_type, max_nprim(shells), max_l(shells), 0);

  // pass operator params to the engines
  engines[0].set_params(oparams);
  for (int i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  const auto shell2bf = map_shell_to_basis_function(shells);

  // Function to compute the integrals in parallel
  auto compute = [&](int thread_id) {
    // buf[0] points to the target shell set after every call  to
    // engines.compute()
    const auto &buf = engines[thread_id].results();

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over
    // Hermitian operators: (1|2) = (2|1)
    for (int s1 = 0; s1 != static_cast<int>(shells.size()); ++s1) {

      int bf1 = shell2bf[s1]; // first basis function in this shell
      int n1 = shells[s1].size();

      for (int s2 = 0; s2 <= s1; ++s2) {
        // Select integrals for current thread
        int acc = s2 + s1 * shells.size();
        if (acc % nthreads != thread_id)
          continue;

        int bf2 = shell2bf[s2];
        int n2 = shells[s2].size();

        // compute shell pair
        engines[thread_id].compute(shells[s1], shells[s2]);

        for (int op = 0; op != nopers; ++op) {
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

/**
 * \brief Read a basis set from HDF5 as a matrix
 */
CP2K_Basis_Atom read_basis_from_hdf5(const string &path_file,
                                     const string &symbol,
                                     const string &basis) {
  libint2::svector<libint2::svector<double>> small_exp;
  libint2::svector<libint2::svector<double>> small_coef;
  libint2::svector<libint2::svector<CP2K_Contractions>> small_fmt;

  // Open an existing HDF5 File
  const File file(path_file, File::ReadOnly);

  // build paths to the coefficients and exponents
  const int n_elec = valence_electrons.at(symbol);
  const string root = "cp2k/basis/" + symbol + "/" + basis + "-q" + std::to_string(n_elec);

  const Group group = file.getGroup(root);
  const std::vector<string> dset_names = group.listObjectNames();

  // Iterate over all exponent sets; for most basis sets there is
  // only a single set of exponents, but there are exception such
  // as BASIS_ADMM_MOLOPT
  for (const auto &name : dset_names) {
    std::vector<std::vector<double>> coefficients;
    std::vector<double> exponents;
    std::vector<int64_t> format;

    const string path_coefficients = root + "/" + name + "/coefficients";
    const string path_exponents = root + "/" + name + "/exponents";

    // Get the dataset
    const DataSet dataset_cs = file.getDataSet(path_coefficients);
    const DataSet dataset_es = file.getDataSet(path_exponents);
    const Attribute attr = dataset_cs.getAttribute("basisFormat");

    // extract data from the datasets
    dataset_cs.read(coefficients);
    dataset_es.read(exponents);
    attr.read(format);

    // Move data to small vectors and keep appending or extending them as
    // iteration over `dset_names` continues
    libint2::svector<double> small_exp_1d;
    std::move(exponents.begin(), exponents.end(), std::back_inserter(small_exp_1d));
    small_exp.push_back(small_exp_1d);

    for (const auto &v : coefficients) {
      libint2::svector<double> small_coef_1d;
      std::move(v.begin(), v.end(), std::back_inserter(small_coef_1d));
      small_coef.push_back(small_coef_1d);
    }

    // The CP2K basis format is defined by a vector of integers, for each atom.
    // For example For the C atom and the Basis DZVP-MOLOPT-GTH the basis format
    // is:
    //  2 0 2 7 2 2 1
    // where:
    //   * 2 is the Principal quantum number
    //   * 0 is the minimum angular momemtum l
    //   * 2 is the maximum angular momentum l
    //   * 7 is the number of total exponents
    //   * 2 Contractions of the 0 + l_min (S) Gaussian Orbitals
    //   * 2 Contractions of the 1 + l_min (P) Gaussian Orbitals
    //   * 1 Contractions of the 2 + l_min (D) Gaussian Orbitals
    //
    // Note: Elements 4 and onwards define the number of contracted for each
    // angular momentum quantum number (all prior elements are disgarded).
    int l, i;
    libint2::svector<CP2K_Contractions> small_fmt_1d;
    for (i=4, l=format[1]; i != static_cast<int>(format.size()); i++, l++) {
      int count = format[i];
      small_fmt_1d.push_back({l, count});
    }
    small_fmt.push_back(small_fmt_1d);
  }
  return CP2K_Basis_Atom{symbol, small_coef, small_exp, small_fmt};
}

/**
 * \brief Return a set of unique symbols
 */
std::vector<string> get_unique_symbols(const std::vector<Atom> &atoms) {
  // Return a set of unique symbols
  std::vector<int> elements;
  std::transform(
    atoms.begin(),
    atoms.end(),
    std::back_inserter(elements),
    [](const Atom &at) { return at.atomic_number; }
  );

  // Unique set of elements
  const std::unordered_set<int> set(elements.begin(), elements.end());

  // create a unique vector of symbols
  std::vector<string> symbols;
  std::transform(
    set.cbegin(),
    set.cend(),
    std::back_inserter(symbols),
    [](const int x) { return map_elements.at(x); }
  );
  return symbols;
}

std::unordered_map<string, CP2K_Basis_Atom>
create_map_symbols_basis(const string &path_hdf5,
                         const std::vector<Atom> &atoms,
                         const string &basis) {
  // Function to generate a map from symbols to basis specification

  std::unordered_map<string, CP2K_Basis_Atom> dict;

  // Select the unique atomic symbols
  const std::vector<string> symbols = get_unique_symbols(atoms);
  for (const auto &at : symbols)
    dict[at] = read_basis_from_hdf5(path_hdf5, at, basis);
  return dict;
}

/**
 * \brief Create the shell specification for a given atom.
 */
libint2::svector<Shell> create_shells_for_atom(const CP2K_Basis_Atom &data,
                                               const Atom &atom) {
  libint2::svector<Shell> shells;
  libint2::svector<double> exponents;
  libint2::svector<CP2K_Contractions> basis_format;

  int acc = 0;
  for (int i = 0; i != static_cast<int>(data.exponents.size()); i++) {
    exponents = data.exponents[i];
    basis_format = data.basis_format[i];
    for (auto &contractions : basis_format) {
      for (int j = 0; j < contractions.count; j++) {
        shells.push_back({
          exponents,
          {{contractions.l, true, data.coefficients[acc]}},  // compute integrals in sphericals
          {{atom.x, atom.y, atom.z}}  // Atomic Coordinates
        });
        acc += 1;
      }
    }
  }
  return shells;
}

/**
 * \brief Make the shell for a CP2K specific basis
 */
std::vector<Shell> make_cp2k_basis(const std::vector<Atom> &atoms,
                                   const string &path_hdf5,
                                   const string &basis) {
  std::vector<Shell> shells;

  // Read basis set data from the HDF5
  const std::unordered_map<string, CP2K_Basis_Atom> dict =
      create_map_symbols_basis(path_hdf5, atoms, basis);

  for (const auto &atom : atoms) {
    const CP2K_Basis_Atom data = dict.at(map_elements.at(atom.atomic_number));
    const auto xs = create_shells_for_atom(data, atom);
    shells.insert(shells.end(), xs.begin(), xs.end());
  }

  return shells;
}
Matrix compute_integrals_couplings(const string &path_xyz_1,
                                   const string &path_xyz_2,
                                   const string &path_hdf5,
                                   const string &basis_name) {

  set_nthread();
  const std::vector<Atom> mol_1 = read_xyz_from_file(path_xyz_1);
  const std::vector<Atom> mol_2 = read_xyz_from_file(path_xyz_2);

  const auto shells_1 = make_cp2k_basis(mol_1, path_hdf5, basis_name);
  const auto shells_2 = make_cp2k_basis(mol_2, path_hdf5, basis_name);

  // safe to use libint now
  libint2::initialize();

  // compute Overlap integrals
  auto S = compute_overlaps_for_couplings(shells_1, shells_2);

  // stop using libint2
  libint2::finalize();

  return S;
}

/**
 * \brief Compute the center of mass for atoms
 */
std::array<double, 3> calculate_center_of_mass(const std::vector<Atom> &atoms) {
  int n = atoms.size();

  std::array<double, 3> rs{0, 0, 0};
  for (auto i = 0; i < n; i++) {
    auto at = atoms[i];
    auto z = double(at.atomic_number);
    rs[0] += at.x * z;
    rs[1] += at.y * z, rs[2] += at.z * z;
  }

  auto m = std::accumulate(
    atoms.begin(),
    atoms.end(),
    0,
    [](const double acc, const Atom &at) { return acc + double(at.atomic_number); }
  );
  return {rs[0] / m, rs[1] / m, rs[2] / m};
}

/**
 * \brief compute the given multipole.
 */
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
  else
    throw std::runtime_error("Unkown multipole");
}

/**
 * \brief   Compute the overlap integrals for the molecule define in `path_xyz`
 * using the `basis_name`
 */
Matrix compute_integrals_multipole(const string &path_xyz,
                                   const string &path_hdf5,
                                   const string &basis_name,
                                   const string &multipole) {
  set_nthread();
  const std::vector<Atom> mol = read_xyz_from_file(path_xyz);

  const auto shells = make_cp2k_basis(mol, path_hdf5, basis_name);

  // safe to use libint now
  libint2::initialize();

  // compute Overlap integrals
  const auto matrices = select_multipole(mol, shells, multipole);

  // stop using libint2
  libint2::finalize();

  Matrix super_matrix(matrices[0].rows() * matrices.size(), matrices[0].cols());
  for (int op = 0; op != static_cast<int>(matrices.size()); ++op) {
    int i = op * matrices[0].rows();
    super_matrix.block(i, 0, matrices[op].rows(), matrices[op].cols()) =
        matrices[op];
  }

  return super_matrix;
}


/** \brief Convert python `str` instance into a C++ std::string **/
int py_obj_to_string(PyObject *py_str, void *ptr) {
  Py_ssize_t size;
  char *str;

  // PyUnicode_AsUTF8AndSize is part of the stable ABI ever since Python 3.10.
  // For older versions we have to use a workaround by first converting the
  // python str instance into a python bytes instance.
#if defined(PyUnicode_AsUTF8AndSize)
  str = PyUnicode_AsUTF8AndSize(py_str, &size);
  if (str == nullptr) {
    return 0;
  }
#else
  PyObject *py_bytes = PyUnicode_AsEncodedString(py_str, "utf8", nullptr);
  if (py_bytes == nullptr) {
    return 0;
  }

  const int num = PyBytes_AsStringAndSize(py_bytes, &str, &size);
  Py_DecRef(py_bytes);
  if (num == -1) {
    return 0;
  }
#endif
  *(string *)ptr = std::string(str, size);
  return 1;
}


/** \brief Convert (copy) an Eigen matrix into a numpy array **/
PyObject *mat_to_npy_array(Matrix &mat) {
  PyObject *npy_array;
  PyObject *ret;
  const npy_intp shape [2] = {mat.rows(), mat.cols()};

  npy_array = PyArray_SimpleNewFromData(2, shape, NPY_DOUBLE, mat.data());
  if (npy_array == nullptr) {
    return nullptr;
  }

  // TODO: Figure out how to transfer ownership of `mat`s memory to the numpy array.
  // As a stop-gap measure, simply create a copy to ensure the arrays' memory is
  // fully managed by numpy.
  ret = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
  int res = PyArray_CopyInto((PyArrayObject *)ret, (PyArrayObject *)npy_array);
  Py_DecRef(npy_array);
  return (res == -1) ? nullptr : ret;
}


/** \brief Python wrapper for compute_integrals_couplings */
PyObject *py_compute_integrals_couplings(PyObject *self, PyObject *args) {
  string path_xyz_1;
  string path_xyz_2;
  string path_hdf5;
  string basis_name;
  Matrix mat;

  if (!PyArg_ParseTuple(args, "O&O&O&O&:compute_integrals_couplings",
      py_obj_to_string, &path_xyz_1,
      py_obj_to_string, &path_xyz_2,
      py_obj_to_string, &path_hdf5,
      py_obj_to_string, &basis_name)) {
    return nullptr;
  }

  try {
    mat = compute_integrals_couplings(path_xyz_1, path_xyz_2, path_hdf5, basis_name);
  } catch (HighFive::Exception &err) {
    PyErr_SetString(PyExc_RuntimeError, err.what());
    return nullptr;
  }
  return mat_to_npy_array(mat);
}


/** \brief Python wrapper for compute_integrals_multipole */
PyObject *py_compute_integrals_multipole(PyObject *self, PyObject *args) {
  string path_xyz;
  string path_hdf5;
  string basis_name;
  string multipole;
  Matrix mat;

  if (!PyArg_ParseTuple(args, "O&O&O&O&:compute_integrals_multipole",
      py_obj_to_string, &path_xyz,
      py_obj_to_string, &path_hdf5,
      py_obj_to_string, &basis_name,
      py_obj_to_string, &multipole)) {
    return nullptr;
  }

  try {
    mat = compute_integrals_multipole(path_xyz, path_hdf5, basis_name, multipole);
  } catch (HighFive::Exception &err) {
    PyErr_SetString(PyExc_RuntimeError, err.what());
    return nullptr;
  }
  return mat_to_npy_array(mat);
}


/** \brief Get the number of threads */
PyObject * py_get_thread_count(PyObject *self, PyObject *args) {
  using libint2::nthreads;
  return PyLong_FromLong(std::thread::hardware_concurrency());
}


/** \brief Get the type of threads */
PyObject * py_get_thread_type(PyObject *self, PyObject *args) {
  PyObject *ret;

#if defined(_OPENMP)
  ret = PyUnicode_FromString("OpenMP");
#else
  ret = PyUnicode_FromString("C++11");
#endif
  return ret;
}


PyMethodDef method_defs[] = {
  {"compute_integrals_couplings", py_compute_integrals_couplings, METH_VARARGS,
    "Compute the basis-set-specific overlap integrals for the molecule as provided in the xyz file."},
  {"compute_integrals_multipole", py_compute_integrals_multipole, METH_VARARGS,
    "Compute the given multipole."},
  {"get_thread_count", py_get_thread_count, METH_NOARGS, "Get the number of threads."},
  {"get_thread_type", py_get_thread_type, METH_NOARGS, "Get the type of threads."},
  {nullptr}  // Sentinel
};


PyModuleDef py_module = {
  PyModuleDef_HEAD_INIT,
  "nanoqm.compute_integrals",                                              // m_name
  "C++ extension module for computing integral overlaps and multipoles.",  // m_doc
  -1,                                                                      // m_size
  method_defs                                                              // m_methods
};

// Workaround to prevent C++ name mangling
//
// The `PyMODINIT_FUNC` macro should be able to take care of it, but it doesn't
// seem to work on python <= 3.8.
// Possibly related to the "-fvisibility=hidden" flag?

#if defined(WIN32) || defined(_WIN32)
  #define MODULE_EXPORT __declspec(dllexport)
#else
  #define MODULE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" MODULE_EXPORT PyObject *PyInit_compute_integrals(void) {
  import_array();
  return PyModule_Create(&py_module);
}
