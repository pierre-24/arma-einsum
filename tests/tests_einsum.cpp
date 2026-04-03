#include <arma_einsum.hpp>

#include <iostream>

#include "tests.hpp"

class ParserTests : public AETestsSuite {};

TEST_F(ParserTests, dot) {
  arma::Col<TEST_FLOAT> a = arma::linspace(1, 5, 5);
  arma::Col<TEST_FLOAT> b = arma::linspace(1, 5, 5);

  std::cout << armaeinsum::einsum_mat<TEST_FLOAT>("i,i->", a, b) << std::endl;
}

TEST_F(ParserTests, trace) {
  arma::Mat<TEST_FLOAT> A = arma::eye(5, 5);

  std::cout << armaeinsum::einsum_mat<TEST_FLOAT>("ii->", A) << std::endl;
}

TEST_F(ParserTests, daxpy) {
  arma::Mat<TEST_FLOAT> A = arma::eye(5, 5);
  arma::Col<TEST_FLOAT> b = arma::linspace(1, 5, 5);

  std::cout << armaeinsum::einsum_mat<TEST_FLOAT>("ik,k->i", A, b) << std::endl;
}

TEST_F(ParserTests, matmul) {
  arma::Mat<TEST_FLOAT> A = arma::eye(5, 5);

  std::cout << armaeinsum::einsum_mat<TEST_FLOAT>("ik,kj->ij", A, A) << std::endl;
}
