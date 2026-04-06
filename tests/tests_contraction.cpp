#include <armadillo>

#define ARMA_EINSUM_DEBUG
#include <arma_einsum.hpp>

#include <string>
#include <vector>

#include "tests.hpp"

class ContractionTests : public AETestsSuite {};

TEST_F(ContractionTests, dot) {
    auto a = arma::randn<arma::Col<TEST_FLOAT>>(5);
    auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

    std::string eq = "i,i->";

    EXPECT_NEAR(
      armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, a, b).at(0, 0),
      arma::dot(a, b),
      1e-5);
}

TEST_F(ContractionTests, trace) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

    std::string eq = "ii->";

    EXPECT_NEAR(
      armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, A).at(0, 0),
      arma::trace(A),
      1e-5);
}

TEST_F(ContractionTests, axpy) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

    std::string eq = "ij,j";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, A, b),
      A * b,
      "abstol", 1e-5));
}

TEST_F(ContractionTests, gemm) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

    std::string eq = "ik,kj";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, A, B),
      A * B,
      "abstol", 1e-5));
}

TEST_F(ContractionTests, hadamard) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

    std::string eq = "ij,ij->ij";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, A, B),
      A % B,
      "abstol", 1e-5));
}

TEST_F(ContractionTests, unitary_transformation) {
    auto R = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    R = R + R.t();

    auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);

    std::string eq = "ji,jk,kl->il";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, R, B, R),
      R.t() * B * R,
      "abstol", 1e-5));
}

TEST_F(ContractionTests, aotomo) {
    // Example taken from quantum chemistry: AO->MO
    auto C = arma::randn<arma::Mat<TEST_FLOAT>>(5, 8);
    auto A_ao = arma::randn<arma::Mat<TEST_FLOAT>>(8, 8);

    std::string eq = "ij,kl,jl->ik";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, C, C, A_ao),
      C * A_ao * C.t(),
      "abstol", 1e-5));
}

TEST_F(ContractionTests, density) {
    // Example taken from quantum chemistry: Density matrix
    auto C = arma::randn<arma::Mat<TEST_FLOAT>>(5, 8);
    auto n = arma::Col<TEST_FLOAT>(5);

    for (auto i = 0; i < 3; i++) {
        n.at(i) = 2.0;
    }

    std::string eq = "i,ik,il->kl";

    EXPECT_TRUE(arma::approx_equal(
      armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, eq, n, C, C),
      C.t() * arma::diagmat(n) * C,
      "abstol", 1e-5));
}

TEST_F(ContractionTests, check_equiv_one_op) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    auto a = arma::randn<arma::Col<TEST_FLOAT>>(5);

    struct ops {
        std::string eq;
        arma::Mat<TEST_FLOAT> opA;
    };

    std::vector<ops> tests = {
        {"i->", a},
        {"i->i", a},
        {"ii", A},
        {"ij->", A},
        {"ji->ij", A},
    };

    for (auto& test : tests) {
        EXPECT_TRUE(arma::approx_equal(
          armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, test.eq, test.opA),
          armaeinsum::einsum_mat<TEST_FLOAT>(test.eq, test.opA),
          "abstol", 1e-5));
    }
}

TEST_F(ContractionTests, check_equiv_two_ops) {
    auto A = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    auto B = arma::randn<arma::Mat<TEST_FLOAT>>(5, 5);
    auto a = arma::randn<arma::Col<TEST_FLOAT>>(5);
    auto b = arma::randn<arma::Col<TEST_FLOAT>>(5);

    struct ops {
        std::string eq;
        arma::Mat<TEST_FLOAT> opA;
        arma::Mat<TEST_FLOAT> opB;
    };

    std::vector<ops> tests = {
        // vec & vec
        {"i,i->", a, b},
        {"i,j", a, b},
        {"i,j->", a, b},
        // vec & mat
        {"ij,j", A, b},
        {"ji,j", A, b},
        {"j,ij", b, A},
        {"i,ij", b, A},
        // mat & mat
        {"ik,kj", A, B},
        {"ki,kj", A, B},
        {"ki,jk", A, B},
        {"ik,jk", A, B},
    };

    for (auto& test : tests) {
        EXPECT_TRUE(arma::approx_equal(
          armaeinsum::einsum_mat_opt<TEST_FLOAT>(armaeinsum::Greedy, test.eq, test.opA, test.opB),
          armaeinsum::einsum_mat<TEST_FLOAT>(test.eq, test.opA, test.opB),
          "abstol", 1e-5));
    }
}
