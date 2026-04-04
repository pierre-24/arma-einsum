#ifndef ARMA_EINSUM_HPP_
#define ARMA_EINSUM_HPP_

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <format>
#include <map>

namespace armaeinsum {

template <typename T>
concept ArmadilloType = arma::is_arma_type<T>::value ||  arma::is_arma_cube_type<T>::value;

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
  : _fixed_values(fixed), _total(1) {
    if (indices.empty()) {
      _total = 1;
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

  uint64_t total() const { return _total; }

  /// Get current value (with fixed)
  multival_t convert(uint64_t counter) const {
    assert(counter < _total);

    multival_t result = _fixed_values;
    uint64_t last_index = _indices.size() - 1;

    uint64_t i = counter;

    for (uint64_t ri = 0; ri < _indices.size(); ri++) {
      result[_indices.at(last_index - ri)] = i % _max_per_index.at(last_index - ri);
      i /= _max_per_index.at(last_index - ri);
    }

    return result;
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
  // 1. Basic validation of result dimensionality
  const auto& result_indices = _eq.back();
  if (result_indices.size() > 2) {
    throw EvaluationError("Result rank > 2; use evaluate_cube() instead.");
  }

  // 2. Validate operand count
  if (sizeof...(Types) != _eq.size() - 1) {
    throw EvaluationError(std::format("Expected {} operands, got {}", _eq.size() - 1, sizeof...(Types)));
  }

  multival_t indices_size;

  // 3. Define and check sizes
  auto check_and_map = [&]<std::size_t... I>(std::index_sequence<I...>) {
    auto validate = [&]<typename T0>(auto idx, const T0& op) {
      using OpType = std::decay_t<T0>;
      const auto& op_labels = _eq[idx];

      // Determine rank at compile-time to avoid "no member" errors
      size_t actual_rank = 0;
      if constexpr (arma::is_arma_cube_type<OpType>::value) {
        actual_rank = 3;
      } else if constexpr (arma::is_Col<OpType>::value || arma::is_Row<OpType>::value) {
        actual_rank = 1;
      } else {
        actual_rank = 2;  // Standard Mat
      }

      if (op_labels.size() != actual_rank) {
        throw EvaluationError(std::format("Rank mismatch for operand #{}", idx));
      }

      // Map index labels to actual Armadillo dimensions
      for (size_t d = 0; d < actual_rank; ++d) {
        char label = op_labels[d];
        uint64_t d_size = 0;

        if (d == 0) {
          d_size = op.n_rows;
        } else if (d == 1) {
          d_size = op.n_cols;
        } else if constexpr (arma::is_arma_cube_type<OpType>::value) {
          d_size = op.n_slices;
        }

        if (indices_size.contains(label)) {
          if (indices_size[label] != d_size) {
            throw EvaluationError(std::format("Size mismatch for index '{}'", label));
          }
        } else {
          indices_size[label] = d_size;
        }
      }
    };  // NOLINT
    (validate(I, operands), ...);
  };  // NOLINT

  check_and_map(std::make_index_sequence<sizeof...(Types)>{});

  // 4. Initialize result matrix
  arma::Mat<T> result;
  if (result_indices.empty()) {
    result.set_size(1, 1);
  } else if (result_indices.size() == 1) {
    result.set_size(indices_size.at(result_indices[0]), 1);
  } else {
    result.set_size(indices_size.at(result_indices[0]), indices_size.at(result_indices[1]));
  }
  result.zeros();  // ensure memory is zeroed

  // 5. Evaluation
  if (result_indices.empty()) {
    IndicesIterator it(indices_size);
    T accum = 0;

#pragma omp parallel for reduction(+:accum)
    for (uint64_t ix=0; ix < it.total(); ++ix) {
      T product = 1;
      multival_t v = it.convert(ix);

      // Perform multiplication across all operands
      auto multiply_operands = [&]<std::size_t... I>(std::index_sequence<I...>) {
        (
            [&](auto idx, const auto& op) {
              using OpType = std::decay_t<decltype(op)>;
              const auto& labels = _eq[idx];

              if constexpr (arma::is_arma_cube_type<OpType>::value) {
                product *= op.at(v.at(labels[0]), v.at(labels[1]), v.at(labels[2]));
              } else if constexpr (arma::is_Col<OpType>::value || arma::is_Row<OpType>::value) {
                product *= op.at(v.at(labels[0]));
              } else {
                product *= op.at(v.at(labels[0]), v.at(labels[1]));
              }
            }(I, operands), ...);
      };  // NOLINT
      multiply_operands(std::make_index_sequence<sizeof...(Types)>{});
      accum += product;
    }
    result.at(0, 0) = accum;
  } else if (result_indices.size() == 1) {
    char index0 = result_indices.at(0);

    for (uint64_t irow = 0; irow < indices_size.at(index0); irow++) {
      T accum = 0;
      IndicesIterator it(indices_size, {{index0, irow}});

#pragma omp parallel for reduction(+:accum)
      for (uint64_t ix=0; ix < it.total(); ++ix) {
        T product = 1;
        multival_t v = it.convert(ix);

        // Perform multiplication across all operands
        auto multiply_operands = [&]<std::size_t... I>(std::index_sequence<I...>) {
          (
              [&](auto idx, const auto& op) {
                using OpType = std::decay_t<decltype(op)>;
                const auto& labels = _eq[idx];

                if constexpr (arma::is_arma_cube_type<OpType>::value) {
                  product *= op.at(v.at(labels[0]), v.at(labels[1]), v.at(labels[2]));
                } else if constexpr (arma::is_Col<OpType>::value || arma::is_Row<OpType>::value) {
                  product *= op.at(v.at(labels[0]));
                } else {
                  product *= op.at(v.at(labels[0]), v.at(labels[1]));
                }
              }(I, operands), ...);
        };  // NOLINT
        multiply_operands(std::make_index_sequence<sizeof...(Types)>{});
        accum += product;
      }

      result.at(irow) = accum;
    }
  } else {
    char index0 = result_indices.at(0), index1 = result_indices.at(1);

    for (uint64_t irow = 0; irow < indices_size.at(index0); irow++) {
      for (uint64_t icol = 0; icol < indices_size.at(index1); icol++) {
        T accum = 0;
        IndicesIterator it(indices_size, {{index0, irow}, {index1, icol}});

#pragma omp parallel for reduction(+:accum)
        for (uint64_t ix=0; ix < it.total(); ++ix) {
          T product = 1;
          multival_t v = it.convert(ix);

          // Perform multiplication across all operands
          auto multiply_operands = [&]<std::size_t... I>(std::index_sequence<I...>) {
            (
                [&](auto idx, const auto& op) {
                  using OpType = std::decay_t<decltype(op)>;
                  const auto& labels = _eq[idx];

                  if constexpr (arma::is_arma_cube_type<OpType>::value) {
                    product *= op.at(v.at(labels[0]), v.at(labels[1]), v.at(labels[2]));
                  } else if constexpr (arma::is_Col<OpType>::value || arma::is_Row<OpType>::value) {
                    product *= op.at(v.at(labels[0]));
                  } else {
                    product *= op.at(v.at(labels[0]), v.at(labels[1]));
                  }
                }(I, operands), ...);
          };  // NOLINT
          multiply_operands(std::make_index_sequence<sizeof...(Types)>{});
          accum += product;
        }

        result.at(irow, icol) = accum;
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
