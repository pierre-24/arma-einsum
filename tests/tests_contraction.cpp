#include <armadillo>
#include <arma_einsum.hpp>

#include <string>

#include "tests.hpp"

class ContractionTests : public AETestsSuite {};

TEST_F(ContractionTests, dot) {
    auto a = arma::randn<arma::Col<TEST_FLOAT>>(5);
    auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

    std::string eq = "i,i->";

    EXPECT_NEAR(
      armaeinsum::ContractionEngine().evaluate_mat<TEST_FLOAT>(armaeinsum::parse(eq), a, b).at(0, 0),
      arma::dot(a, b),
      1e-5);
}

TEST_F(ContractionTests, trace) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

    std::string eq = "ii->";

    EXPECT_NEAR(
      armaeinsum::ContractionEngine().evaluate_mat<TEST_FLOAT>(armaeinsum::parse(eq), A).at(0, 0),
      arma::trace(A),
      1e-5);
}

TEST_F(ContractionTests, axpy) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

    std::string eq = "ij,j";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::ContractionEngine().evaluate_mat<TEST_FLOAT>(armaeinsum::parse(eq), A, b),
      A * b,
      "abstol", 1e-5));
}

TEST_F(ContractionTests, gemm) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

    std::string eq = "ik,kj";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::ContractionEngine().evaluate_mat<TEST_FLOAT>(armaeinsum::parse(eq), A, B),
      A * B,
      "abstol", 1e-5));
}

TEST_F(ContractionTests, unitary_transformation) {
    auto R = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    R = R + R.t();

    auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

    std::string eq = "ji,jk,kl->il";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::ContractionEngine().evaluate_mat<TEST_FLOAT>(armaeinsum::parse(eq), R, B, R),
      R.t() * B * R,
      "abstol", 1e-5));
}
