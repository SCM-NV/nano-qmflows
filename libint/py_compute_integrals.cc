/*
 * This module contains the implementation of several
 * kind of integrals used for non-adiabatic molecular dynamics,
 * including the overlaps integrals between different geometries
 * And the dipoles and quadrupoles to compute absorption spectra.
 * This module is based on libint and Eigen.
 * Copyright (C) 2018-2022 the Netherlands eScience Center.
 */

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x03080000
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "namd.hpp"
#include "compute_integrals.hpp"

using std::string;
using namd::Matrix;


/** \brief Convert python `str` instance into a C++ std::string **/
int py_str_to_string(PyObject *py_str, void *ptr) {
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
  *(string *)ptr = string(str, size);
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
      py_str_to_string, &path_xyz_1,
      py_str_to_string, &path_xyz_2,
      py_str_to_string, &path_hdf5,
      py_str_to_string, &basis_name)) {
    return nullptr;
  }

  try {
    mat = compute_integrals_couplings(path_xyz_1, path_xyz_2, path_hdf5, basis_name);
  } catch (std::exception &err) {
    return PyErr_SetFromErrno(PyExc_RuntimeError);
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
      py_str_to_string, &path_xyz,
      py_str_to_string, &path_hdf5,
      py_str_to_string, &basis_name,
      py_str_to_string, &multipole)) {
    return nullptr;
  }

  try {
    mat = compute_integrals_multipole(path_xyz, path_hdf5, basis_name, multipole);
  } catch (std::exception &err) {
    return PyErr_SetFromErrno(PyExc_RuntimeError);
  }
  return mat_to_npy_array(mat);
}


/** \brief Get the number of threads */
PyObject *py_get_thread_count(PyObject *self, PyObject *args) {
  return PyLong_FromUnsignedLong(std::thread::hardware_concurrency());
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
// The `PyMODINIT_FUNC` macro should make all required symbols visible,
// but it doesn't seem to work on python <= 3.8.
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
