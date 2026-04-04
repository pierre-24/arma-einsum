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


using multival_t = std::map<char, uint64_t>;

/**
 * Iterator over multiple indices
 */
class IndicesIterator {
 private:
  /// list of indices
  std::vector<char> _indices;
  /// List of max
  std::vector<uint64_t> _max_per_index;
  /// Fixed values, added as is to the result
  multival_t _fixed_values;
  /// Internal counter
  uint64_t _counter;
  /// Internal total, computed as `prod(_max_per_index)`
  uint64_t _total;

 public:
  IndicesIterator() = delete;

  /**
   * Create an iterator
   *
   * @param indices list indices and their max value
   * @param fixed set of extra indices which has a fixed value (added as is to the result)
   */
  explicit IndicesIterator(const multival_t& indices, const multival_t& fixed = {})
  : _fixed_values(fixed), _counter(0), _total(1) {
    if (indices.empty()) {
      _total = 0;
    } else {
      for (auto& index : indices) {
        if (!fixed.contains(index.first)) {
          _indices.push_back(index.first);
          _max_per_index.push_back(index.second);
          _total *= index.second;
        }
      }
    }
  }

  /// Is there a next value?
  [[nodiscard]] bool has_next() const {
    return _counter < _total;
  }

  /// Get current value (with fixed)
  multival_t operator*() const {
    multival_t result = _fixed_values;
    uint64_t last_index = _indices.size() - 1;

    uint64_t i = _counter;

    for (uint64_t ri = 0; ri < _indices.size(); ri++) {
      result[_indices.at(last_index - ri)] = i % _max_per_index.at(last_index - ri);
      i /= _max_per_index.at(last_index - ri);
    }

    return result;
  }

  /// Iterate
  void next() {
    if (has_next()) {
      _counter++;
    }
  }
};

using indices_t = std::vector<char>;

/**
 * Equation
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
  multival_t indices_size;

  auto define_and_check = [&]<typename T0>(uint64_t& iop_, const T0 & op) {
    std::vector<uint64_t> op_dims;
    if (arma::is_Col<std::decay_t<T0>>::value) {
      op_dims.push_back(op.n_rows);
    } else if (arma::is_Row<std::decay_t<T0>>::value) {
      op_dims.push_back(op.n_rows);
    }  else {  // assume a matrix (and hope for the best)
      op_dims.push_back(op.n_rows);
      op_dims.push_back(op.n_cols);
    }

    const auto& op_indices = _eq.at(iop_);

    if (op_dims.size() != op_indices.size()) {
      throw EvaluationError(
        std::format("operand size ({}) and actual operand size ({}) mismatch for operand #{}",
          op_dims.size(), op_indices.size(), iop_));
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

    iop_++;
  };  // NOLINT

  uint64_t iop = 0;
  (define_and_check(iop, operands), ...);

  // set result size & evaluate
  arma::Mat<T> result(1, 1);

  auto evaluate_mat = [&]<typename T0>(uint64_t& iop_, T& val, const multival_t& v_, const T0& op) {
    const auto& op_indices = _eq.at(iop_);

    if (arma::is_Col<std::decay_t<T0>>::value) {
      val *= op.at(v_.at(op_indices.at(0)));
    } else if (arma::is_Row<std::decay_t<T0>>::value) {
      val *= op.at(v_.at(op_indices.at(0)));
    }  else {  // assume a matrix (and hope for the best)
      val *= op.at(v_.at(op_indices.at(0)), v_.at(op_indices.at(1)));
    }

    iop_++;
  };  // NOLINT

  if (result_indices.empty()) {
    IndicesIterator it(indices_size);
    while (it.has_next()) {
      auto val = *it;
      T tmp = 1;
      iop = 0;
      (evaluate_mat(iop, tmp, val, operands), ...);
      result.at(0, 0) += tmp;
      it.next();
    }
  } else if (result_indices.size() == 1) {
    char index0 = result_indices.at(0);
    if (!indices_size.contains(index0)) {
      throw EvaluationError(std::format("unknown index `{}` for result", index0));
    }
    result.resize(indices_size.at(index0));

    for (uint64_t irow = 0; irow < indices_size.at(index0); irow++) {
      IndicesIterator it(indices_size, {{index0, irow}});
      while (it.has_next()) {
        auto val = *it;
        T tmp = 1;
        iop = 0;
        (evaluate_mat(iop, tmp, val, operands), ...);
        result.at(irow) += tmp;
        it.next();
      }
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

    for (uint64_t irow = 0; irow < indices_size.at(index0); irow++) {
      for (uint64_t icol = 0; icol < indices_size.at(index1); icol++) {
        IndicesIterator it(indices_size, {{index0, irow}, {index1, icol}});
        while (it.has_next()) {
          auto val = *it;
          T tmp = 1;
          iop = 0;
          (evaluate_mat(iop, tmp, val, operands), ...);
          result.at(irow, icol) += tmp;
          it.next();
        }
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

  // if not, use non-repeated indices in order
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
