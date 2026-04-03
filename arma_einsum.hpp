#ifndef ARMA_EINSUM_HPP_
#define ARMA_EINSUM_HPP_

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <tuple>
#include <map>

#include <armadillo>

namespace armaeinsum {

template <typename T>
concept ArmadilloType = arma::is_arma_type<T>::value;

class ParserError: public std::runtime_error {
 private:
  uint64_t _position;
 public:
  ParserError() = delete;
  explicit ParserError(uint64_t position, const std::string& msg) :
    runtime_error(std::format("error at position {}: {}", position, msg)), _position(position) {}
};

class EvaluationError: public std::runtime_error {
 public:
  EvaluationError() = delete;
  explicit EvaluationError(const std::string& msg) : runtime_error(msg) {}
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

  uint64_t n() const { return _indices.size(); }

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

  uint64_t n() const { return _operands.size(); }

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

  /**
   * Evaluate the equation using `operands` and return a matrix
   *
   * @tparam T a floating point type
   * @tparam Types armadillo type
   * @param operands operands, must match `n()`
   * @return a matrix defined by the equation
   */
  template <typename T, ArmadilloType... Types> arma::Mat<T> evaluate_mat(const Types&... operands);
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
    throw ParserError(position, std::format("too many indices ({}), armadillo only can go up to 3", indices.size()));
  }

  return Indices(indices);
}

template <typename T, ArmadilloType... Types>
arma::Mat<T> Equation::evaluate_mat(const Types&... operands) {
  // check if result is a matrix
  if (_result.n() > 2) {
    throw EvaluationError("use evaluate_cube() for result.n() > 2");
  }

  // check size of operands
  size_t num_operands = sizeof...(operands);
  if (num_operands != _operands.size()) {
    throw EvaluationError(
      std::format("number of operands ({}) do not match definition ({})", num_operands, _operands.size()));
  }

  // get a tuple
  auto pack_tuple = std::forward_as_tuple(operands...);

  // specify and check the type of each array in operands
  std::map<char, uint64_t> indices_size;
  uint64_t iop = 0;

  auto define_and_check = [&]<typename T0>(const T0 & op) {
    std::vector<uint64_t> op_dims;
    if (arma::is_Col<std::decay_t<T0>>::value) {
      op_dims.push_back(op.n_rows);
    } else if (arma::is_Row<std::decay_t<T0>>::value) {
      op_dims.push_back(op.n_rows);
    }  else {  // assume a matrix (and hope for the best)
      op_dims.push_back(op.n_rows);
      op_dims.push_back(op.n_cols);
    }

    const auto& op_indices = _operands.at(iop).indices();

    if (op_dims.size() != op_indices.size()) {
      throw EvaluationError(
        std::format("operand size ({}) and actual operand size ({}) mismatch for operand #{}",
          op_dims.size(), op_indices.size(), iop));
    }

    for (uint64_t i = 0; i < op_dims.size(); i++) {
      char index = op_indices.at(i);
      if (indices_size.contains(index)) {
        if (indices_size[index] != op_dims.at(i)) {
          throw EvaluationError(
            std::format("size mismatch for operand `{}` ({}!={})",
              index, indices_size[index], op_dims.at(i)));
        }
      } else {
        indices_size[index] = op_dims.at(i);
      }
    }

    iop++;
  };  // NOLINT

  (define_and_check(operands), ...);

  // set result size & evaluate
  arma::Mat<T> result(1, 1);

  if (_result.n() == 0) {
  } else if (_result.n() == 1) {
    char index0 = _result.indices().at(0);
    if (!indices_size.contains(index0)) {
      throw EvaluationError(std::format("unknown index `{}` for result", index0));
    }
    result.resize(indices_size.at(index0));

#pragma omp parallel for
    for (uint64_t irow = 0; irow < indices_size.at(index0); irow++) {
      result.at(irow) = 0;
    }

  } else if (_result.n() == 2) {
    char index0 = _result.indices().at(0), index1 = _result.indices().at(1);
    if (!indices_size.contains(index0)) {
      throw EvaluationError(std::format("unknown index `{}` for result", index0));
    }
    if (!indices_size.contains(index1)) {
      throw EvaluationError(std::format("unknown index `{}` for result", index1));
    }
    result.resize(indices_size.at(index0), indices_size.at(index1));

#pragma omp parallel for
    for (uint64_t irow = 0; irow < indices_size.at(index0); irow++) {
      for (uint64_t icol = 0; icol < indices_size.at(index1); icol++) {
        result.at(irow, icol) = 0;
      }
    }
  }

  return result;
}

/**
 * Parse a given equation
 *
 * @param equation the equation, a string containing `EQUATION` (defined in class `Equation`)
 * @return a valid `Equation` if `equation` is a valid string
 */
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

template <typename T, ArmadilloType... Types>
arma::Mat<T> einsum_mat(const std::string& equation, const Types&... operands) {
  return parse(equation).evaluate_mat<T>(operands...);
}

}  // namespace armaeinsum


#endif  // ARMA_EINSUM_HPP_
