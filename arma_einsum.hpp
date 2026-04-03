#ifndef ARMA_EINSUM_HPP_
#define ARMA_EINSUM_HPP_

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

namespace armaeinsum {

class ParserError: public std::runtime_error {
 private:
  uint64_t _position;
 public:
  ParserError() = delete;
  explicit ParserError(uint64_t position, const std::string& msg) : runtime_error(msg), _position(position) {}
};

/**
 * Expression node
 */
class ExprNode {
 public:
  ExprNode() = default;
  virtual ~ExprNode() = default;

  /// string representation
  virtual explicit operator std::string() const = 0;
};

/**
 * Indices node
 *
 * INDICES := [a-zA-Z]{0,3}
 */
class Indices : public ExprNode {
 private:
  std::vector<char> _indices;
 public:
  Indices() = default;

  explicit Indices(const std::vector<char>& indices) : ExprNode(), _indices(indices) {}

  const std::vector<char>& indices() { return _indices; }

  explicit operator std::string() const override {
    std::string result;
    result.resize(_indices.size());
    std::copy(_indices.begin(), _indices.end(), result.begin());

    return result;
  }
};

/**
 * Equation node
 *
 * EQ := INDICES (',' INDICES)* ('->' INDICES)?
 *
 */
class Equation: public ExprNode {
 private:
  std::vector<Indices> _operands;
  Indices _result;

 public:
  Equation();

  Equation(const std::vector<Indices>& operands, Indices result):
    ExprNode(), _operands(operands), _result(std::move(result)) {}

  explicit operator std::string() const override {
    std::stringstream ss;

    for (uint64_t i = 0; i < _operands.size(); i++) {
      if (i > 0) {
        ss << ",";
      }

      ss << std::string(_operands.at(i));
    }

    ss << "->" << std::string(_result);

    return ss.str();
  }
};

inline Indices _parse_indices(const std::string& input, uint64_t& position) {
  if (position >= input.length()) {
    throw ParserError(position, "expected indices, got EOS");
  }

  if (!isalpha(input[position])) {
    throw ParserError(position, "expected a letter for indices");
  }

  std::vector<char> indices;
  while (position < input.length() && isalpha(input[position])) {
    indices.push_back(input[position]);
    position++;
  }

  if (indices.empty()) {
    throw ParserError(position, "expected at least one index");
  }

  if (indices.size() > 3) {
    throw ParserError(position, "too many indices, armadillo only can go up to 3");
  }

  return Indices(indices);
}

inline Equation parse(const std::string& equation) {
  if (equation.empty()) {
    throw ParserError(0, "empty equation");
  }

  uint64_t i = 0;

  // parse operands
  std::vector<Indices> operands;
  while (i < equation.length()) {
    if (isalpha(equation[i])) {
      operands.push_back(_parse_indices(equation, i));
    } else if (equation[i] == ',') {
      i++;
    } else if (equation[i] == '-') {
      break;
    } else {
      throw ParserError(i, "unexpected character");
    }
  }

  if (operands.empty()) {
    throw ParserError(i, "expected at least one operand");
  }

  // parse result if any
  if (equation[i] == '-') {
    i++;
    if (equation[i] != '>') {
      throw ParserError(i, "expected '>'");
    }
    i++;

    if (i == equation.length()) {  // result is a single number
      return {operands, Indices()};
    }

    // there are indices for result
    Indices result = _parse_indices(equation, i);

    if (i != equation.length()) {
      throw ParserError(i, "expected EOS, but the string is longer");
    }

    return {operands, result};
  }

  // if not, non-repeated indices in order
  std::vector<char> nri;
  for (auto& operand : operands) {
    for (const auto& c : operand.indices()) {
      if (auto position = std::find(nri.begin(), nri.end(), c); position != nri.end()) {
        nri.erase(position);
      } else {
        nri.push_back(c);
      }
    }
  }

  return {operands, Indices(nri)};
}

}  // namespace armaeinsum


#endif  // ARMA_EINSUM_HPP_
