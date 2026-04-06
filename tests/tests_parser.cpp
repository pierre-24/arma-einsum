#include <armadillo>
#include <arma_einsum.hpp>

#include <string>

#include "tests.hpp"

class ParserTests : public AETestsSuite {};

TEST_F(ParserTests, SimpleTest) {
  // dot product
  auto eq = armaeinsum::Equation::parse("a,a");
  EXPECT_EQ("a,a->", std::string(eq));

  // trace
  eq = armaeinsum::Equation::parse("ii");
  EXPECT_EQ("ii->", std::string(eq));

  // keep as is
  eq = armaeinsum::Equation::parse("ij");
  EXPECT_EQ("ij->ij", std::string(eq));

  // transpose (automatic with reordering)
  eq = armaeinsum::Equation::parse("ji");
  EXPECT_EQ("ji->ij", std::string(eq));

  // sum
  eq = armaeinsum::Equation::parse("ij->");
  EXPECT_EQ("ij->", std::string(eq));

  // matrix product
  eq = armaeinsum::Equation::parse("ik,kj");
  EXPECT_EQ("ik,kj->ij", std::string(eq));
}
