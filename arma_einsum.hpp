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
 public:
  ParserError() = delete;
  explicit ParserError(uint64_t position, const std::string& msg) :
    runtime_error(std::format("error at position {}: {}", position, msg)) {}
};

class EvaluationError: public std::runtime_error {
 public:
  EvaluationError() = delete;
  explicit EvaluationError(const std::string& msg) : runtime_error(msg) {}
};

using indices_t = std::vector<char>;

/**
 * Equation
 *
 */
class Equation {
 private:
  std::vector<indices_t> _eq;

 public:
  Equation();

  explicit Equation(const std::vector<indices_t>& indices): _eq(indices) {
    if (indices.size() < 2) {
      throw EvaluationError("not enough operands");
    }

    // check if there are enough pair of indices
    indices_t nri;
    for (auto& operand : _eq) {
      for (const auto& c : operand) {
        if (auto position = std::find(nri.begin(), nri.end(), c); position != nri.end()) {
          nri.erase(position);
        } else {
          nri.push_back(c);
        }
      }
    }

    if (!nri.empty()) {
      std::stringstream ss;
      for (const auto& c : nri) {
        ss << c << " ";
      }

      throw EvaluationError(std::format("non-paired indices: {}", ss.str()));
    }
  }

  [[nodiscard]] uint64_t n() const { return _eq.size(); }

  explicit operator std::string() const {
    std::stringstream ss;

    for (uint64_t i = 0; i < _eq.size() - 1; i++) {
      if (i > 0) {
        ss << ",";
      }

      for (auto& c : _eq[i]) {
        ss << c;
      }
    }

    ss << "->";

    for (auto& c : _eq[_eq.size() - 1]) {
      ss << c;
    }

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

template <typename T, ArmadilloType... Types>
arma::Mat<T> Equation::evaluate_mat(const Types&... operands) {
  // check if result is a matrix
  auto& result_indices = _eq[_eq.size() - 1];

  if (result_indices.size() > 2) {
    throw EvaluationError("use evaluate_cube() for result.n() > 2");
  }

  // check size of operands
  size_t num_operands = sizeof...(operands);
  if (num_operands != _eq.size() - 1) {
    throw EvaluationError(
      std::format("number of operands ({}) do not match definition ({})", num_operands, _eq.size() - 1));
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

    const auto& op_indices = _eq.at(iop);

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

  if (result_indices.empty()) {
  } else if (result_indices.size() == 1) {
    char index0 = result_indices.at(0);
    if (!indices_size.contains(index0)) {
      throw EvaluationError(std::format("unknown index `{}` for result", index0));
    }
    result.resize(indices_size.at(index0));

#pragma omp parallel for
    for (uint64_t irow = 0; irow < indices_size.at(index0); irow++) {
      result.at(irow) = 0;
    }

  } else if (result_indices.size() == 2) {
    char index0 = result_indices.at(0), index1 = result_indices.at(1);
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

inline indices_t _parse_indices(const std::string& input, uint64_t& position) {
  if (position >= input.length()) {
    throw ParserError(position, "expected indices, got EOS");
  }

  if (!isalpha(input[position])) {
    throw ParserError(position, "expected a letter for indices");
  }

  indices_t indices;
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

  return indices;
}

/**
 * Parse a given equation:
 *
 * EQUATION := INDICES (',' INDICES)* ('->' INDICES)?
 * INDICES := [a-zA-Z]{0,3}
 *
 * @param equation the equation
 * @return a valid `Equation` if `equation` is a valid string
 */
inline Equation parse(const std::string& equation) {
  if (equation.empty()) {
    throw ParserError(0, "empty equation");
  }

  uint64_t i = 0;

  // parse operands
  std::vector<indices_t> eq;
  while (i < equation.length()) {
    if (isalpha(equation[i])) {
      eq.push_back(_parse_indices(equation, i));
    } else if (equation[i] == ',') {
      i++;
    } else if (equation[i] == '-') {
      break;
    } else {
      throw ParserError(i, "unexpected character");
    }
  }

  if (eq.empty()) {
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
      eq.push_back(indices_t());
      return Equation(eq);
    }

    // there are indices for result
    eq.push_back(_parse_indices(equation, i));

    if (i != equation.length()) {
      throw ParserError(i, "expected EOS, but the string is longer");
    }

    return Equation(eq);
  }

  // if not, non-repeated indices in order
  indices_t nri;
  for (auto& operand : eq) {
    for (const auto& c : operand) {
      if (auto position = std::find(nri.begin(), nri.end(), c); position != nri.end()) {
        nri.erase(position);
      } else {
        nri.push_back(c);
      }
    }
  }

  if (nri.size() > 3) {
    throw ParserError(i, std::format("too many indices ({}) in result, armadillo only can go up to 3", nri.size()));
  }

  eq.push_back(nri);

  return Equation(eq);
}

template <typename T, ArmadilloType... Types>
arma::Mat<T> einsum_mat(const std::string& equation, const Types&... operands) {
  return parse(equation).evaluate_mat<T>(operands...);
}

}  // namespace armaeinsum


#endif  // ARMA_EINSUM_HPP_
