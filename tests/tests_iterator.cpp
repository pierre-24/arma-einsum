#include <armadillo>
#include <arma_einsum.hpp>

#include <vector>

#include "tests.hpp"

class IteratorTests : public AETestsSuite {};

TEST_F(IteratorTests, test_iterator) {
  armaeinsum::IndicesIterator it({{'a', 2}, {'b', 3}, {'c',  2}}, {'a', 'b', 'c'});
  bool set[2][3][2] = {false};

  while (it.has_next()) {
    auto r = *it;
    set[r[0]][r[1]][r[2]]  = true;
    ++it;
  }

  for (uint64_t i = 0; i < 2; i++) {
    for (uint64_t j = 0; j < 3; j++) {
      for (uint64_t k = 0; k < 2; k++) {
        EXPECT_TRUE(set[i][j][k]);
      }
    }
  }
}

TEST_F(IteratorTests, test_iterator_with_fixed) {
  armaeinsum::IndicesIterator it({{'a', 2}, {'b', 3}}, {'a', 'b', 'c'}, {{'c', 1}});
  bool set[2][3][2] = {false};

  while (it.has_next()) {
    auto r = *it;
    set[r[0]][r[1]][r[2]]  = true;
    ++it;
  }

  for (uint64_t i = 0; i < 2; i++) {
    for (uint64_t j = 0; j < 3; j++) {
      for (uint64_t k = 0; k < 2; k++) {
        if (k == 0) {
          EXPECT_FALSE(set[i][j][k]);
        } else {
          EXPECT_TRUE(set[i][j][k]);
        }
      }
    }
  }
}

TEST_F(IteratorTests, test_iterator_no_loop) {
  armaeinsum::IndicesIterator it({}, {'c'}, {{'c', 1}});

  EXPECT_EQ(*it, std::vector<uint64_t>({1}));
  EXPECT_EQ(it.max(), 1);
}
