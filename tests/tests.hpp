#ifndef TESTS_TESTS_HPP_
#define TESTS_TESTS_HPP_

#include <gtest/gtest.h>

#define TEST_FLOAT double

class AETestsSuite : public testing::Test {
 public:
  AETestsSuite() = default;
};

#endif  // TESTS_TESTS_HPP_
