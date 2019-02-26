#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// integrals library
#include <libint2.hpp>

// #include <pybind11/pybind11.h>

// Eigen matrix algebra library
#include <Eigen/Dense>

// HDF5 funcionality
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

using std::string;
using HighFive::Attribute;
using HighFive::File;
using HighFive::DataSet;
using libint2::Atom;
using libint2::Engine;
using libint2::Operator;
using libint2::Shell;

using real_t = libint2::scalar_type;

// import dense, dynamically sized Matrix type from Eigen;
// this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
// to meet the layout of the integrals returned by the Libint integral library
using  Matrix = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// compute the number of basis set for the molecule
size_t nbasis(const std::vector<libint2::Shell>& shells);

// Count the number of basis functions for each shell
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells);

// Overlaps for the derivative coupling
Matrix compute_overlaps(const std::vector<Shell>& shells_1, const std::vector<Shell>& shells_2);

int compute_integrals(const string& path_xyz, const string& basis_name);
libint2::BasisSet create_basis_set(const string& basis_name, const std::vector<Atom>& atoms);

void test_read(const string& path_hdf5, const string& path_xyz, const string& basis);

struct CP2K_Basis_Atom {
  // Contains the basis specificationf for a given atom
  string symbol;
  std::vector<std::vector<double>> coefficients;
  std::vector<double> exponents;
  std::vector<int> basis_format;
};

int main() {

  string path_xyz = "../test/test_files/ethylene.xyz";
  string path_hdf5 = "../test/test_files/C.hdf5";
  string basis_name = "6-311g**";
  // string basis_name = "sto-3g";

  auto xs = compute_integrals(path_xyz, basis_name);
  test_read(path_hdf5, path_xyz, "DZVP-MOLOPT-SR-GTH");
    }


int compute_integrals(const string& path_xyz, const string& basis_name) {
  // Compute the overlap integrals for the molecule define in `path_xyz` using
  // the `basis_name`
  
  // Read molecular geometry
  std::ifstream input_file(path_xyz);
  std::vector<Atom> atoms = libint2::read_dotxyz(input_file);

  auto shells = create_basis_set(basis_name, atoms);

  // safe to use libint now
  libint2::initialize();

  // compute Overlap integrals
  auto S = compute_overlaps(shells, shells);
  std::cout << "rows: " << S.rows() << " cols: " << S.cols() << "\n";
  // std::cout << "\n\tOverlap Integrals:\n";
  // std::cout << S << "\n";
  
  libint2::finalize();
  
  return 42;
}

// PYBIND11_MODULE(call_libint, m) {
//     m.doc() = "Compute integrals using libint2 see: https://github.com/evaleev/libint/wiki";

//     m.def("compute_integrals", &compute_integrals, "Compute integrals using libint2");
// }


size_t nbasis(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (const auto& shell: shells)
    n += shell.size();
  return n;
}

auto max_nprim(const std::vector<libint2::Shell>& shells) {
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
  Engine engine(Operator::overlap, max_nprim(shells_1), max_l(shells_1), 0);
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

libint2::BasisSet create_basis_set(const string& basis_name, const std::vector<Atom>& atoms) {
  // Create a basis set of non-standard basis set for CP2K
  
  return libint2::BasisSet{basis_name, atoms};
}



// // Function to read the basis set from the HDF5
// CP2K_Basis_Atom read_basis_from_hdf5(const string& path_hdf5, const string& symbol, const string& basis);


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
  std::unordered_map<int, string> z_element = {
    {1, "h"}, {2, "he"}, {3, "li"}, {4, "be"}, {5, "b"}, {6, "c"}, {7, "n"}, {8, "o"},
    {9, "f"}, {10, "ne"}, {11, "na"}, {12, "mg"}, {13, "al"}, {14, "si"}, {15, "p"}, {16, "s"},
    {17, "cl"}, {18, "ar"}, {19, "k"}, {20, "ca"}, {21, "sc"}, {22, "ti"}, {23, "v"}, {24, "cr"},
    {25, "mn"}, {26, "fe"}, {27, "co"}, {28, "ni"}, {29, "cu"}, {30, "zn"}, {31, "ga"}, {32, "ge"},
    {33, "as"}, {34, "se"}, {35, "br"}, {36, "kr"}, {37, "rb"}, {38, "sr"}, {39, "y"}, {40, "zr"},
    {41, "nb"}, {42, "mo"}, {43, "tc"}, {44, "ru"}, {45, "rh"}, {46, "pd"}, {47, "ag"}, {48, "cd"},
    {49, "in"}, {50, "sn"}, {51, "sb"}, {52, "te"}, {53, "i"}, {54, "xe"}, {55, "cs"}, {56, "ba"},
    {57, "la"}, {58, "ce"}, {59, "pr"}, {60, "nd"}, {61, "pm"}, {62, "sm"}, {63, "eu"}, {64, "gd"},
    {65, "tb"}, {66, "dy"}, {67, "ho"}, {68, "er"}, {69, "tm"}, {70, "yb"}, {71, "lu"}, {72, "hf"},
    {73, "ta"}, {74, "w"}, {75, "re"}, {76, "os"}, {77, "ir"}, {78, "pt"}, {79, "au"}, {80, "hg"},
    {81, "tl"}, {82, "pb"}, {83, "bi"}, {84, "po"}, {85, "at"}, {86, "rn"}, {87, "fr"}, {88, "ra"},
    {89, "ac"}, {90, "th"}, {91, "pa"}, {92, "u"}, {93, "np"}, {94, "pu"}, {95, "am"}, {96, "cm"}
  };
  std::vector<int> elements;
  std::transform(atoms.begin(), atoms.end(), std::back_inserter(elements),
		 [](const Atom& at) {return at.atomic_number;}
		 );
  // Unique set of elements
  std::unordered_set<int> set(elements.begin(), elements.end());

  // create a unique vector of symbols
  std::vector<string> symbols;
  std::transform(set.cbegin(), set.cend(), std::back_inserter(symbols),
		 [&z_element](int x) {return z_element[x];});
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


std::vector<Shell> make_cp2k_basis(const std::vector<Atom>& atoms, const string& Basis) {
  // Make the shell for a CP2K specific basis

    std::vector<Shell> shells;

    
    
//     for(auto at=0; a<atoms.size(); ++at) {

//       shells.push_back({
// 	  exponents,
// 	    {
// 	      {contraction, false, coefficients}
// 	    },
// 	      {{atoms[at].x, atoms[at].y, atoms[at].z}}   // origin coordinates
// 	}
// 	);	    
//     }
      
    return shells;
}

void test_read(const string& path_hdf5, const string& path_xyz, const string& basis) {
  std::ifstream input_file(path_xyz);
  std::vector<Atom> atoms = libint2::read_dotxyz(input_file);
  std::vector<string> symbols = get_unique_symbols(atoms);

 
  std::unordered_map<string, CP2K_Basis_Atom> result =
    create_map_symbols_basis(path_hdf5, atoms, basis);

  for (const auto & pair: result){
    std::cout << "element: " << pair.first << "\n";
    std::cout << "exponents:" << "\n";
    for (auto x: pair.second.exponents)
      std::cout << x << " ";
    std::cout << "\n";
    // }
  }
}
