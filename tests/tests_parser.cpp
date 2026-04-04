#include <armadillo>
#include <arma_einsum.hpp>

#include <string>

#include "tests.hpp"

class ParserTests : public AETestsSuite {};

TEST_F(ParserTests, SimpleTest) {
  // dot product
  auto eq = armaeinsum::parse("a,a");
  EXPECT_EQ("a,a->", std::string(eq));

  // trace
  eq = armaeinsum::parse("ii");
  EXPECT_EQ("ii->", std::string(eq));

  // keep as is
  eq = armaeinsum::parse("ij");
  EXPECT_EQ("ij->ij", std::string(eq));

  // sum
  eq = armaeinsum::parse("ij->");
  EXPECT_EQ("ij->", std::string(eq));

  // matrix product
  eq = armaeinsum::parse("ik,kj");
  EXPECT_EQ("ik,kj->ij", std::string(eq));
}
