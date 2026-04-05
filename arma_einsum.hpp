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
#include <map>
#include <set>

#ifndef ARMA_EINSUM_FORMAT  // use std::format by default
#include <format>
#define ARMA_EINSUM_FORMAT std::format
#endif

namespace armaeinsum {

template <typename T>
concept ArmadilloType = arma::is_arma_type<T>::value ||  arma::is_arma_cube_type<T>::value;

class ParserError: public std::runtime_error {
 public:
  ParserError() = delete;
  explicit ParserError(uint64_t position, const std::string& msg) :
    runtime_error(ARMA_EINSUM_FORMAT("error at position {}: {}", position, msg)) {}
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
  /// Maximal value for each id
  std::vector<uint64_t> _max_per_id;
  /// Current value
  std::vector<uint64_t> _current_v;
  /// Active ids
  std::vector<size_t> _active_ids;

  uint64_t _max_val;
  uint64_t _current_ix;

 public:
  IndicesIterator(
    const multival_t& indices, const std::vector<char>& all_labels, const multival_t& fixed = {})
    : _max_val(1), _current_ix(0) {
    size_t num_slots = all_labels.size();
    _max_per_id.assign(num_slots, 1);
    _current_v.assign(num_slots, 0);

    auto get_id = [&](char c) {
      auto it = std::find(all_labels.begin(), all_labels.end(), c);
      return static_cast<size_t>(std::distance(all_labels.begin(), it));
    };

    for (const auto& [label, val] : fixed) {
      _current_v[get_id(label)] = val;
    }

    for (const auto& [label, max_dim] : indices) {
      if (!fixed.contains(label)) {
        size_t id = get_id(label);
        _active_ids.push_back(id);
        _max_per_id[id] = max_dim;
        _max_val *= max_dim;
      }
    }

    std::sort(_active_ids.begin(), _active_ids.end());
  }

  /// Accessor: operator* returns the current vector state
  const std::vector<uint64_t>& operator*() const {
    return _current_v;
  }

  /// Increment
  IndicesIterator& operator++() {  // The "Odometer" step logic
    _current_ix++;
    for (size_t id : _active_ids) {
      _current_v[id]++;
      if (_current_v[id] < _max_per_id[id]) {
        return *this;
      }
      _current_v[id] = 0;
    }
    return *this;
  }

  /// Boolean check for the while loop
  [[nodiscard]] bool has_next() const {
    return _current_ix < _max_val;
  }

  [[nodiscard]] uint64_t max() const { return _max_val; }

  /**
   * start_to / jump_to logic: Sets the state for a specific flat index
   */
  void start_to(uint64_t goal_index) {
    _current_ix = goal_index;
    uint64_t temp = goal_index;
    for (size_t id : _active_ids) {
      _current_v[id] = temp % _max_per_id[id];
      temp /= _max_per_id[id];
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
  template <typename T, ArmadilloType... Types> arma::Mat<T> evaluate_mat(const Types&... operands) const;
};

template <typename T, ArmadilloType... Types>
arma::Mat<T> Equation::evaluate_mat(const Types&... operands) const {
  // 1. Basic validation of result dimensionality
  const auto& result_indices = _eq.back();
  if (result_indices.size() > 2) {
    throw EvaluationError("Result rank > 2; use evaluate_cube() instead.");
  }

  // 2. Validate operand count
  if (sizeof...(Types) != _eq.size() - 1) {
    throw EvaluationError(ARMA_EINSUM_FORMAT("Expected {} operands, got {}", _eq.size() - 1, sizeof...(Types)));
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
        throw EvaluationError(ARMA_EINSUM_FORMAT("Rank mismatch for operand #{}", idx));
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
            throw EvaluationError(ARMA_EINSUM_FORMAT("Size mismatch for index '{}'", label));
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
  result.zeros();

  // Setup: Create the master label list (schema) and the ID maps
  std::vector<char> all_labels;
  std::set<char> unique_labels;
  for (const auto& labels : _eq) {
    for (char c : labels) {
      unique_labels.insert(c);
    }
  }

  all_labels.assign(unique_labels.begin(), unique_labels.end());

  auto get_id = [&](char c) {
    auto it = std::find(all_labels.begin(), all_labels.end(), c);
    return static_cast<size_t>(std::distance(all_labels.begin(), it));
  };

  // Pre-calculate the integer IDs for every operand
  std::vector<std::vector<size_t>> op_id_map;
  for (size_t i = 0; i < sizeof...(Types); ++i) {
    std::vector<size_t> ids;
    for (char c : _eq[i]) {
      ids.push_back(get_id(c));
    }

    op_id_map.push_back(ids);
  }

  auto compute_product = [&]<std::size_t... I>(const std::vector<uint64_t>& v, std::index_sequence<I...>) -> T {
    T product = 1;
    ([&](auto idx, const auto& op) {
      using OpType = std::decay_t<decltype(op)>;
      const auto& ids = op_id_map[idx];

      if constexpr (arma::is_arma_cube_type<OpType>::value) {
        product *= op.at(v[ids[0]], v[ids[1]], v[ids[2]]);
      } else if constexpr (arma::is_Col<OpType>::value || arma::is_Row<OpType>::value) {
        product *= op.at(v[ids[0]]);
      } else {
        product *= op.at(v[ids[0]], v[ids[1]]);
      }
    }(I, operands), ...);
    return product;
  };

  constexpr auto seq = std::make_index_sequence<sizeof...(Types)>{};

  // 5. Evaluate
  if (result_indices.empty()) {  // SCALAR RESULT CASE
    // We create a global iterator just to get the max() value for partitioning
    IndicesIterator total_iterator(indices_size, all_labels);
    uint64_t total_work = total_iterator.max();
    T total_accum = 0;

#pragma omp parallel for reduction(+:total_accum) if (!omp_in_parallel()) \
        default(none) shared(total_work, all_labels, indices_size, seq, compute_product)
    for (int chunk = 0; chunk < omp_get_num_threads(); ++chunk) {
      // Divide work manually
      int tid = omp_get_thread_num();
      int n_threads = omp_get_num_threads();
      uint64_t start = (total_work * tid) / n_threads;
      uint64_t end = (total_work * (tid + 1)) / n_threads;

      IndicesIterator it(indices_size, all_labels);
      it.start_to(start);  // O(Rank) jump once per thread

      T local_accum = 0;
      for (uint64_t i = start; i < end && it.has_next(); ++i) {
        local_accum += compute_product(*it, seq);
        ++it;
      }
      total_accum = local_accum;
    }
    result.at(0, 0) = total_accum;
  } else if (result_indices.size() == 1) {  // VECTOR RESULT CASE
    char idx0 = result_indices[0];

#pragma omp parallel for if (!omp_in_parallel()) \
        default(none) shared(seq, compute_product, result, indices_size, idx0, all_labels, op_id_map)
    for (uint64_t irow = 0; irow < indices_size.at(idx0); ++irow) {
      T accum = 0;
      IndicesIterator it(indices_size, all_labels, {{idx0, irow}});

      while (it.has_next()) {
        accum += compute_product(*it, seq);
        ++it;
      }
      result.at(irow) = accum;
    }
  } else {  // MATRIX RESULT CASE
    char idx0 = result_indices[0], idx1 = result_indices[1];

#pragma omp parallel for collapse(2) if (!omp_in_parallel()) \
        default(none) shared(seq, compute_product, result, indices_size, idx0, idx1, all_labels, op_id_map)
    for (uint64_t irow = 0; irow < indices_size.at(idx0); ++irow) {
      for (uint64_t icol = 0; icol < indices_size.at(idx1); ++icol) {
        T accum = 0;
        IndicesIterator it(indices_size, all_labels, {{idx0, irow}, {idx1, icol}});

        while (it.has_next()) {
          accum += compute_product(*it, seq);
          ++it;
        }
        result.at(irow, icol) = accum;
      }
    }
  }

  return result;
}

class ContractionEngine {
 public:
  ContractionEngine() = default;

  /**
   * Evaluate `eq` by optimzing it if possible
   *
   * @tparam T a floating-point type
   * @tparam Types Armadillo objects
   * @param eq an equation
   * @param operands Armadillo objects
   * @return a matrix with the evaluated result
   */
  template <typename T, ArmadilloType... Types>
  arma::Mat<T> evaluate_mat(const Equation& eq, const Types&... operands) const;
};

template <typename T, ArmadilloType ... Types>
arma::Mat<T> ContractionEngine::evaluate_mat(const Equation& eq, const Types&... operands) const {
  return eq.evaluate_mat<T>(operands...);
}

/// Parse indices
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
    throw ParserError(
      position, ARMA_EINSUM_FORMAT("too many indices ({}), armadillo only can go up to 3", indices.size()));
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
    throw ParserError(
      i, ARMA_EINSUM_FORMAT("too many indices ({}) in result, armadillo only can go up to 3", nri.size()));
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
