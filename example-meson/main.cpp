#include <iostream>

#define ARMA_USE_OPENMP
#include <armadillo>

#include <arma_einsum.hpp>

int main() {
  arma::vec a = arma::linspace(1, 5, 5);
  arma::vec b = arma::linspace(1, 5, 5);
  
  std::cout << armaeinsum::einsum_mat<double>("i,i->", a, b).at(0, 0) << std::endl;  // should print "55"
  return 0;
}
