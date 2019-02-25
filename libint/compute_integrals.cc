#include <fstream>
#include <iostream>
#include <libint2.hpp>
// #include <pybind11/pybind11.h>
#include <string>
#include <vector>

// Eigen matrix algebra library
#include <Eigen/Dense>

using std::string;
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

int main() {

  string path_xyz = "../test/test_files/ethylene.xyz";
  string basis_name = "6-311g**";
  // string basis_name = "sto-3g";

  auto xs = compute_integrals(path_xyz, basis_name);
  
    }


int compute_integrals(const string& path_xyz, const string& basis_name) {
  // Compute the overlap integrals for the molecule define in `path_xyz` using
  // the `basis_name`
  
  // Read molecular geometry
  std::ifstream input_file(path_xyz);
  std::vector<Atom> atoms = libint2::read_dotxyz(input_file);

  // Create Basis
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


// std::vector<Shell> make_cp2k_basis(const std::vector<Atom>& atoms) {
//   // Make the shell for a CP2K specific basis

//     std::vector<Shell> shells;

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
      
//     return shells;
// }
