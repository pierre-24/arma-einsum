#include <armadillo>
#include <arma_einsum.hpp>

#include "tests.hpp"

class ParserTests : public AETestsSuite {};

TEST_F(ParserTests, dot) {
  auto a = arma::randn<arma::Col<TEST_FLOAT>>(5);
  auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("i,i->", a, b).at(0, 0),
    arma::dot(a, b),
    1e-5);
}

TEST_F(ParserTests, outer) {
  auto a = arma::randn<arma::Col<TEST_FLOAT>>(5);
  auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("i,j", a, b),
    a * b.t(),
    "abstol", 1e-5));
}

TEST_F(ParserTests, trace) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("ii->", A).at(0, 0),
    arma::trace(A),
    1e-5);
}

TEST_F(ParserTests, sum) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("ij->", A).at(0, 0),
    arma::accu(A),
    1e-5);
}

TEST_F(ParserTests, transpose) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ij->ji", A),
    A.t(),
    "abstol", 1e-5));
}

TEST_F(ParserTests, axpy) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
  auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ik,k->i", A, b),
    A * b,
    "abstol", 1e-5));
}

TEST_F(ParserTests, gemm) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
  auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ik,kj->ij", A, B),
    A * B,
    "abstol", 1e-5));
}

TEST_F(ParserTests, orthogonal) {
  auto R = arma::randn<arma::Mat<TEST_FLOAT>>(8, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ji,jk->ik", R, R),
    R.t() * R,
    "abstol", 1e-5));
}

TEST_F(ParserTests, unitary_transformation) {
  auto R = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
  R = R + R.t();

  auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ji,jk,kl->il", R, B, R),
    R.t() * B * R,
    "abstol", 1e-5));
}

TEST_F(ParserTests, traces_with_cube) {
  auto A = arma::randn<arma::Cube<TEST_FLOAT>>(5, 5, 2);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("iij", A).at(0),
    arma::trace(A.slice(0)),
    1e-5);
}

TEST_F(ParserTests, xsum_with_cubes) {
  auto A = arma::randn<arma::Cube<TEST_FLOAT>>(5, 5, 2);
  auto B = arma::randn<arma::Cube<TEST_FLOAT>>(5, 5, 2);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("jki,jkl", A, B).at(0, 0),
    armaeinsum::einsum_mat<TEST_FLOAT>("jk,jk", A.slice(0), B.slice(0)).at(0, 0),
    1e-5);
}
