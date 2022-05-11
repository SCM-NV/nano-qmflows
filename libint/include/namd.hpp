/*
 * This module contains the implementation of several
 * kind of integrals used for non-adiabatic molecular dynamics,
 * including the overlaps integrals between different geometries
 * And the dipoles and quadrupoles to compute absorption spectra.
 * This module is based on libint and Eigen.
 * Copyright (C) 2018-2022 the Netherlands eScience Center.
 */

#ifndef NAMD_H_
#define NAMD_H_

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

// Eigen matrix algebra library
#include <Eigen/Dense>

// HDF5 funcionality
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace namd {

using real_t = libint2::scalar_type;
// import dense, dynamically sized Matrix type from Eigen;
// this is a matrix with row-major storage
// (http://en.wikipedia.org/wiki/Row-major_order) to meet the layout of the
// integrals returned by the Libint integral library
using Matrix =
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct CP2K_Contractions {
    int l;  // Angular momentum quantum number for a given shell-type
    int count;  // Number of contractions for a given shell-type
};

struct CP2K_Basis_Atom {
  // Contains the basis specificationf for a given atom
  std::string symbol;
  libint2::svector<libint2::svector<double>> coefficients;
  libint2::svector<libint2::svector<double>> exponents;
  libint2::svector<libint2::svector<CP2K_Contractions>> basis_format;
};

// Map from atomic_number to symbol
const std::unordered_map<int, std::string> map_elements = {
    {1, "h"},   {2, "he"},  {3, "li"},  {4, "be"},  {5, "b"},   {6, "c"},
    {7, "n"},   {8, "o"},   {9, "f"},   {10, "ne"}, {11, "na"}, {12, "mg"},
    {13, "al"}, {14, "si"}, {15, "p"},  {16, "s"},  {17, "cl"}, {18, "ar"},
    {19, "k"},  {20, "ca"}, {21, "sc"}, {22, "ti"}, {23, "v"},  {24, "cr"},
    {25, "mn"}, {26, "fe"}, {27, "co"}, {28, "ni"}, {29, "cu"}, {30, "zn"},
    {31, "ga"}, {32, "ge"}, {33, "as"}, {34, "se"}, {35, "br"}, {36, "kr"},
    {37, "rb"}, {38, "sr"}, {39, "y"},  {40, "zr"}, {41, "nb"}, {42, "mo"},
    {43, "tc"}, {44, "ru"}, {45, "rh"}, {46, "pd"}, {47, "ag"}, {48, "cd"},
    {49, "in"}, {50, "sn"}, {51, "sb"}, {52, "te"}, {53, "i"},  {54, "xe"},
    {55, "cs"}, {56, "ba"}, {57, "la"}, {58, "ce"}, {59, "pr"}, {60, "nd"},
    {61, "pm"}, {62, "sm"}, {63, "eu"}, {64, "gd"}, {65, "tb"}, {66, "dy"},
    {67, "ho"}, {68, "er"}, {69, "tm"}, {70, "yb"}, {71, "lu"}, {72, "hf"},
    {73, "ta"}, {74, "w"},  {75, "re"}, {76, "os"}, {77, "ir"}, {78, "pt"},
    {79, "au"}, {80, "hg"}, {81, "tl"}, {82, "pb"}, {83, "bi"}, {84, "po"},
    {85, "at"}, {86, "rn"}, {87, "fr"}, {88, "ra"}, {89, "ac"}, {90, "th"},
    {91, "pa"}, {92, "u"},  {93, "np"}, {94, "pu"}, {95, "am"}, {96, "cm"}};

// Map from atomic symbols to the number of valence electrons
// See also `nanoqm.workflows.templates.valence_electrons`
const std::unordered_map<std::string, int> valence_electrons = {
    {"ag", 11},
    {"al", 3},
    {"ar", 8},
    {"as", 5},
    {"at", 7},
    {"au", 11},
    {"b", 3},
    {"ba", 10},
    {"be", 4},
    {"bi", 5},
    {"br", 7},
    {"c", 4},
    {"ca", 10},
    {"cd", 12},
    {"cl", 7},
    {"co", 17},
    {"cr", 4},
    {"cs", 9},
    {"cu", 11},
    {"f", 7},
    {"fe", 16},
    {"ga", 13},
    {"ge", 4},
    {"h", 1},
    {"he", 2},
    {"hf", 12},
    {"hg", 12},
    {"i", 7},
    {"in", 13},
    {"ir", 17},
    {"k", 9},
    {"kr", 8},
    {"li", 3},
    {"mg", 10},
    {"mn", 15},
    {"mo", 14},
    {"n", 5},
    {"na", 9},
    {"nb", 13},
    {"ne", 8},
    {"ni", 18},
    {"o", 6},
    {"os", 16},
    {"p", 5},
    {"pb", 4},
    {"pd", 18},
    {"po", 6},
    {"pt", 18},
    {"rb", 9},
    {"re", 15},
    {"rh", 17},
    {"rn", 8},
    {"ru", 16},
    {"s", 6},
    {"sb", 5},
    {"sc", 11},
    {"se", 6},
    {"si", 4},
    {"sn", 4},
    {"sr", 10},
    {"ta", 13},
    {"tc", 15},
    {"te", 6},
    {"ti", 12},
    {"tl", 13},
    {"u", 14},
    {"v", 13},
    {"w", 14},
    {"xe", 8},
    {"y", 11},
    {"zn", 12},
    {"zr", 12},
};

} // namespace namd
#endif // NAMD_H_
