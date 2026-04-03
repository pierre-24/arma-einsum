#include <arma_einsum.hpp>

#include <string>

#include "tests.hpp"

class ParserTests : public AETestsSuite {};

TEST_F(ParserTests, SimpleTest) {
  // dot product
  auto eq = armaeinsum::parse("a,a");
  ASSERT_EQ("a,a->", std::string(eq));

  // trace
  eq = armaeinsum::parse("ii");
  ASSERT_EQ("ii->", std::string(eq));

  // matrix product
  eq = armaeinsum::parse("ik,kj");
  ASSERT_EQ("ik,kj->ij", std::string(eq));
}
