#include <armadillo>

#define ARMA_EINSUM_DEBUG
#include <arma_einsum.hpp>

#include <string>
#include <vector>

#include "tests.hpp"

class ContractionTests : public AETestsSuite {};

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
