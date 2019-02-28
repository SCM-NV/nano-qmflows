#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <vector>

// integrals library
#include <libint2.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// Eigen matrix algebra library
#include <Eigen/Dense>

// HDF5 funcionality
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

// Constants
#include "namd.h"

namespace py = pybind11;
using std::string;
using HighFive::Attribute;
using HighFive::File;
using HighFive::DataSet;
using libint2::Atom;
using libint2::Shell;
using namd::CP2K_Basis_Atom;
using namd::map_elements;
using namd::Matrix;


Matrix compute_integrals(const string& path_xyz_1,
			 const string& path_xyz_2,
			 const string& path_hdf5,
			 const string& basis_name);

std::vector<Atom> read_xyz_from_file(const string& path_xyz) {
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
    compute_integrals(path_xyz, path_xyz, path_hdf5, basis_name);
    }

size_t nbasis(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (const auto& shell: shells)
    n += shell.size();
  return n;
}

size_t max_nprim(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (auto shell: shells)
    n = std::max(shell.nprim(), n);
  return n;
}

int max_l(const std::vector<libint2::Shell>& shells) {
  int l = 0;
  for (auto shell: shells)
    for (auto c: shell.contr)
      l = std::max(c.l, l);
  return l;
}

std::vector<size_t> map_shell_to_basis_function(const std::vector<Shell>& shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for (auto shell: shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}

Matrix compute_overlaps(const std::vector<Shell>& shells_1, const std::vector<Shell>& shells_2)
{

  const auto n = nbasis(shells_1);
  Matrix result(n, n);

  // construct the overlap integrals engine
  libint2::Engine engine(libint2::Operator::overlap, max_nprim(shells_1), max_l(shells_1), 0);
  auto shell2bf = map_shell_to_basis_function(shells_1);

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // loop over unique shell pairs, {s1,s2}
  // Notem
  for(auto s1=0; s1!=shells_1.size(); ++s1) {

    auto bf1 = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells_1[s1].size();

    for(auto s2=0; s2 != shells_2.size(); ++s2) {

      auto bf2 = shell2bf[s2];
      auto n2 = shells_2[s2].size();

      // compute shell pair
      engine.compute(shells_1[s1], shells_2[s2]);

      // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
      Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
      result.block(bf1, bf2, n1, n2) = buf_mat;
    }
  }

  return result;
}

std::vector<int> read_basisFormat(const string& basisFormat){
  // Transform the string containing the basis format for CP2K, into a vector of strings
  std::vector<int> rs;
  for(auto x: basisFormat){
    if (std::isdigit(x)) 
      // convert char to int
      rs.push_back(x - '0');
  }
  return rs;
}

CP2K_Basis_Atom read_basis_from_hdf5(const string& path_file, const string& symbol, const string& basis) {
  // Read a basis set from HDF5 as a matrix

  std::vector<std::vector<double>> coefficients;
  std::vector<double> exponents;
  string format;
  try{
  
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

  } catch (HighFive::Exception& err) {
    // catch and print any HDF5 error
    std::cerr << err.what() << std::endl;
  }
  return CP2K_Basis_Atom{symbol, coefficients, exponents, read_basisFormat(format)};
}

std::vector<string> get_unique_symbols(const std::vector<Atom>& atoms) {
  // Return a set of unique symbols
  std::vector<int> elements;
  std::transform(atoms.begin(), atoms.end(), std::back_inserter(elements),
		 [](const Atom& at) {return at.atomic_number;}
		 );
  // Unique set of elements
  std::unordered_set<int> set(elements.begin(), elements.end());

  // create a unique vector of symbols
  std::vector<string> symbols;
  std::transform(set.cbegin(), set.cend(), std::back_inserter(symbols),
		 [](int x) {return map_elements[x];});
  return symbols;
}


std::unordered_map<string, CP2K_Basis_Atom> create_map_symbols_basis(
   const string& path_hdf5, const std::vector<Atom>& atoms, const string& basis) {
  // Function to generate a map from symbols to basis specification
  
  std::unordered_map<string, CP2K_Basis_Atom> dict;

  // Select the unique atomic symbols
  std::vector<string> symbols = get_unique_symbols(atoms);
  for (const auto& at: symbols)
    dict[at] = read_basis_from_hdf5(path_hdf5, at, basis);

  return dict;
}

std::vector<Shell> create_shells_for_atom(const CP2K_Basis_Atom& data, const Atom& atom) {
  // Create the shell specification for a given atom.
  // The CP2K basis format is defined by a vector of integers, for each atom. For example
  // For the C atom and the Basis DZVP-MOLOPT-GTH the basis format is:
  //  2 0 2 7 2 2 1
  // where:
  //   * 2 is the Principal quantum number
  //   * 0 is the minimum angular momemtum l
  //   * 2 is the maximum angular momentum l
  //   * 7 is the number of total exponents
  //   * 2 Contractions of S Gaussian Orbitals
  //   * 2 Contractions of P Gaussian Orbitals
  //   * 1 Contraction of D Gaussian Orbital
  // Note: From element 4 onwards are define the number of contracted for each quantum number.
  std::vector<int> basis_format = data.basis_format;
  std::vector<Shell> shells;

  auto acc = 0;
  for (auto i=0; i+4 < basis_format.size(); i++){
    for (auto j=0; j < basis_format[i + 4]; j++){
      shells.push_back({
	  data.exponents,
	    { // compute integrals in sphericals
	      {i, true, data.coefficients[acc]}
	    },
	    // Atomic Coordinates
	      {{atom.x, atom.y, atom.z}}
	}
	);
      acc += 1;
    }
  }
  return shells;
}

std::vector<Shell> make_cp2k_basis(const std::vector<Atom>& atoms, const string& path_hdf5, const string& basis) {
  // Make the shell for a CP2K specific basis
  
  std::vector<Shell> shells;

  // set of symbols
  std::vector<string> symbols = get_unique_symbols(atoms);

  // Read basis set data from the HDF5
  std::unordered_map<string, CP2K_Basis_Atom> dict =
    create_map_symbols_basis(path_hdf5, atoms, basis);
    
  for(const auto& atom: atoms) {

    CP2K_Basis_Atom data = dict[map_elements[atom.atomic_number]];
    auto xs = create_shells_for_atom(data, atom);
    shells.insert(shells.end(), xs.begin(), xs.end());
  }
      
    return shells;
}

std::tuple<string, string> get_group_dataset(const string& path) {

  std::size_t found = path.rfind("/");

  return std::make_tuple(path.substr(0, found), path.substr(found + 1, path.length()));
}

Matrix compute_integrals(const string& path_xyz_1,
			 const string& path_xyz_2,
			 const string& path_hdf5,
			 const string& basis_name) {
  // Compute the overlap integrals for the molecule define in `path_xyz` using
  // the `basis_name`

  std::vector<Atom> mol_1 = read_xyz_from_file(path_xyz_1);
  std::vector<Atom> mol_2 = read_xyz_from_file(path_xyz_2);
  
  auto shells_1 = make_cp2k_basis(mol_1, path_hdf5, basis_name);
  auto shells_2 = make_cp2k_basis(mol_2, path_hdf5, basis_name);

  // safe to use libint now
  libint2::initialize();

  // compute Overlap integrals
  auto S = compute_overlaps(shells_1, shells_2);

  // stop using libint2
  libint2::finalize();
  
  return S;
}

PYBIND11_MODULE(compute_integrals, m) {
    m.doc() = "Compute integrals using libint2 see: https://github.com/evaleev/libint/wiki";

    m.def("compute_integrals", &compute_integrals, py::return_value_policy::reference_internal);
}
