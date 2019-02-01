#include <libint2.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

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
// One body Operator
Matrix compute_1body_ints(const std::vector<Shell>& shells,
                          Operator t,
                          const std::vector<Atom>& atoms = std::vector<Atom>());

int compute_integrals(const string& path_xyz, const string& basis_name);


int main() {

  string path_xyz = "../test/test_files/ethylene.xyz";
  string basis_name = "sto-3g";

  auto xs = compute_integrals(path_xyz, basis_name);
  
    }


int compute_integrals(const string& path_xyz, const string& basis_name) {
  // Compute the overlap integrals for the molecule define in `path_xyz` using
  // the `basis_name`
  
  // Read molecular geometry
  std::ifstream input_file(path_xyz);
  std::vector<Atom> atoms = libint2::read_dotxyz(input_file);

  for (const auto& at: atoms)
    std::cout << "atomic_number: " << at.atomic_number << " coordinates: " << at.x << " "<< at.y << " " << at.z << "\n";

  // Create Basis 
  libint2::BasisSet shells(basis_name, atoms);

  // safe to use libint now
  libint2::initialize();

  // compute Overlap integrals
    auto S = compute_1body_ints(shells, Operator::overlap);
    std::cout << "rows: " << S.rows() << " cols: " << S.cols() << "\n";
    // std::cout << "\n\tOverlap Integrals:\n";
    // std::cout << S << "\n";
  
  libint2::finalize();
  
  std::cout << "all right!" << "\n";
  
  return 42;
}


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

Matrix compute_1body_ints(const std::vector<Shell>& shells,
                          libint2::Operator obtype,
                          const std::vector<Atom>& atoms)
{

  const auto n = nbasis(shells);
  Matrix result(n,n);

  // construct the overlap integrals engine
  Engine engine(obtype, max_nprim(shells), max_l(shells), 0);
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical charges
  if (obtype == Operator::nuclear) {
    std::vector<std::pair<real_t,std::array<real_t,3>>> q;
    for(const auto& atom : atoms) {
      q.push_back( {static_cast<real_t>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
    }
    engine.set_params(q);
  }

  auto shell2bf = map_shell_to_basis_function(shells);

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // loop over unique shell pairs, {s1,s2} such that s1 >= s2
  // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
  for(auto s1=0; s1!=shells.size(); ++s1) {

    auto bf1 = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2 = shell2bf[s2];
      auto n2 = shells[s2].size();

      // compute shell pair
      engine.compute(shells[s1], shells[s2]);

      // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
      Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
      result.block(bf1, bf2, n1, n2) = buf_mat;
      if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
      result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

    }
  }

  return result;
}
