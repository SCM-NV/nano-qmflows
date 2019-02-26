#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

using std::string;
using HighFive::Attribute;
using HighFive::File;
using HighFive::DataSet;

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

auto read_basis_from_hdf5(const string& path_file, const string& symbol, const string& basis) {
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
  return std::make_tuple(coefficients, exponents, read_basisFormat(format));
}
int main() {
  string path_hdf5 = "../test/test_files/C.hdf5";
  string name_dataset = "DZVP-MOLOPT-SR-GTH";
  std::vector<std::vector<double>> coefficients;
  std::vector<double> exponents;
  std::vector<int> basisFormat;

  std::tie(coefficients, exponents, basisFormat) = read_basis_from_hdf5(path_hdf5, "c", name_dataset);
  
  std::cout << "coefficients" << "\n";
  for (auto r: coefficients){
    std::cout << "vector: ";
    for (auto x: r) {
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
  std::cout << "exponents" << "\n";
  for (auto r: exponents){
    std::cout << r << " ";
  }

  std::cout << "basisFormat: ";
  for (const auto& x: basisFormat)
    std::cout << x << " ";
  std::cout << "\n";

  
}
