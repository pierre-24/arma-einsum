#include <armadillo>

#define ARMA_EINSUM_DEBUG
#include <arma_einsum.hpp>

#include <string>

#include "tests.hpp"

class EinsumTests : public AETestsSuite {};

TEST_F(EinsumTests, dot) {
  auto a = arma::randn<arma::Col<TEST_FLOAT>>(5);
  auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("i,i->", a, b).at(0, 0),
    arma::dot(a, b),
    1e-5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "i,i->", a, b).at(0, 0),
    arma::dot(a, b),
    1e-5);
}

TEST_F(EinsumTests, outer) {
  auto a = arma::randn<arma::Col<TEST_FLOAT>>(5);
  auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("i,j", a, b),
    a * b.t(),
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "i,j", a, b),
    a * b.t(),
    "abstol", 1e-5));
}

TEST_F(EinsumTests, trace) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("ii->", A).at(0, 0),
    arma::trace(A),
    1e-5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "ii->", A).at(0, 0),
    arma::trace(A),
    1e-5);
}

TEST_F(EinsumTests, sum) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("ij->", A).at(0, 0),
    arma::accu(A),
    1e-5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "ij->", A).at(0, 0),
    arma::accu(A),
    1e-5);
}

TEST_F(EinsumTests, transpose) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ij->ji", A),
    A.t(),
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "ij->ji", A),
    A.t(),
    "abstol", 1e-5));
}

TEST_F(EinsumTests, axpy) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
  auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ik,k->i", A, b),
    A * b,
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "ik,k->i", A, b),
    A * b,
    "abstol", 1e-5));
}

TEST_F(EinsumTests, gemm) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
  auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ik,kj->ij", A, B),
    A * B,
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "ik,kj->ij", A, B),
    A * B,
    "abstol", 1e-5));
}

TEST_F(EinsumTests, hadamard) {
  auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
  auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ij,ij->ij", A, B),
    A % B,
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "ij,ij->ij", A, B),
    A % B,
    "abstol", 1e-5));
}

TEST_F(EinsumTests, orthogonal) {
  auto R = arma::randn<arma::Mat<TEST_FLOAT>>(8, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ji,jk->ik", R, R),
    R.t() * R,
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "ji,jk->ik", R, R),
    R.t() * R,
    "abstol", 1e-5));
}

TEST_F(EinsumTests, unitary_transformation) {
  auto R = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
  R = R + R.t();

  auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>("ji,jk,kl->il", R, B, R),
    R.t() * B * R,
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "ji,jk,kl->il", R, B, R),
    R.t() * B * R,
    "abstol", 1e-5));
}

TEST_F(EinsumTests, aotomo) {
  // Example taken from quantum chemistry: AO->MO
  auto C = arma::randn<arma::Mat<TEST_FLOAT>>(5, 8);
  auto A_ao = arma::randn<arma::Mat<TEST_FLOAT>>(8, 8);

  std::string eq = "ij,kl,jl->ik";

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>(eq, C, C, A_ao),
    C * A_ao * C.t(),
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, C, C, A_ao),
    C * A_ao * C.t(),
    "abstol", 1e-5));
}

TEST_F(EinsumTests, density) {
  // Example taken from quantum chemistry: Density matrix
  auto C = arma::randn<arma::Mat<TEST_FLOAT>>(5, 8);
  auto n = arma::Col<TEST_FLOAT>(5);

  for (auto i = 0; i < 3; i++) {
    n.at(i) = 2.0;
  }

  std::string eq = "i,ik,il->kl";

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat<TEST_FLOAT>(eq, n, C, C),
    C.t() * arma::diagmat(n) * C,
    "abstol", 1e-5));

  EXPECT_TRUE(arma::approx_equal(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, n, C, C),
    C.t() * arma::diagmat(n) * C,
    "abstol", 1e-5));
}

TEST_F(EinsumTests, traces_with_cube) {
  auto A = arma::randn<arma::Cube<TEST_FLOAT>>(5, 5, 2);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("iij", A).at(0),
    arma::trace(A.slice(0)),
    1e-5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "iij", A).at(0),
    arma::trace(A.slice(0)),
    1e-5);
}

TEST_F(EinsumTests, xsum_with_cubes) {
  auto A = arma::randn<arma::Cube<TEST_FLOAT>>(5, 5, 2);
  auto B = arma::randn<arma::Cube<TEST_FLOAT>>(5, 5, 2);

  EXPECT_NEAR(
    armaeinsum::einsum_mat<TEST_FLOAT>("jki,jkl", A, B).at(0, 0),
    armaeinsum::einsum_mat<TEST_FLOAT>("jk,jk", A.slice(0), B.slice(0)).at(0, 0),
    1e-5);

  EXPECT_NEAR(
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "jki,jkl", A, B).at(0, 0),
    armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, "jk,jk", A.slice(0), B.slice(0)).at(0, 0),
    1e-5);
}
