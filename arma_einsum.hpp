#ifndef ARMA_EINSUM_HPP_
#define ARMA_EINSUM_HPP_

#include <iostream>
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
#include <list>

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
 * Equation, hold the sets of indices (as `_operands`).
 * This object is immutable.
 */
class Equation {
 private:
  std::vector<indices_t> _eq;

  static indices_t _calculate_implicit(const std::vector<indices_t>& operands);

  static void _validate_usage(const std::vector<indices_t>& operands);

 public:
  Equation() = default;
  Equation(const Equation& o) = default;
  Equation(Equation&&) = default;
  Equation& operator=(const Equation&) = default;

  explicit Equation(const std::vector<indices_t>& indices): _eq(indices) {
    _validate_usage(_eq);
  }

  /// Get operands
  [[nodiscard]] const std::vector<indices_t>& operands() const  { return _eq; }

  /// Get a given operand
  [[nodiscard]] const indices_t& at(uint64_t index) const {
    return _eq.at(index);
  }

  /// Give the number of operands (including the resulting one)
  [[nodiscard]] uint64_t n() const { return _eq.size(); }

  /// Give the unique indices
  [[nodiscard]] std::set<char> unique_indices() const {
    std::set<char> uniques;
    for (auto& i : _eq) {
      for (auto& c : i) {
        uniques.insert(c);
      }
    }

    return uniques;
  }

  /// Give the length of the equation (i.e., the sum of the size of each operand
  [[nodiscard]] uint64_t length() const {
    uint64_t len = 0;
    for (auto& i : _eq) {
      len += i.size();
    }
    return len;
  }

  /// pretty-printing
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
   * Evaluate the size of each index, given `operands`
   *
   * @tparam Types Armadillo types
   * @param operands Armadillo objects
   * @return a std::map with, for each index in `_eq`, its size
   */
  template <ArmadilloType... Types> multival_t indices_size(const Types&... operands) const;

  /**
   * Estimate the flop count of `this`.
   *
   * @tparam T A floating point type
   * @param indices_size the size of all indices appearing in `this`
   * @return the estimated flop count to evaluate `this`
   */
  template <typename T> T estimate_flop_count(const multival_t& indices_size);

  /**
   * Evaluate the equation using `operands` and return a matrix
   *
   * @tparam T a floating point type
   * @tparam Types armadillo type
   * @param operands operands, must match `n()`
   * @return a matrix defined by the equation
   */
  template <typename T, ArmadilloType... Types> arma::Mat<T> evaluate_mat(const Types&... operands) const;

  /**
   * Parse a given equation:
   *
   * EQUATION := INDICES (',' INDICES)* ('->' INDICES)?
   * INDICES := [a-zA-Z]{0,3}
   *
   * @param equation the equation
   * @return a valid `Equation` if `equation` is a valid string
   */
  static Equation parse(const std::string& equation);
};

inline indices_t Equation::_calculate_implicit(const std::vector<indices_t>& operands) {
  uint64_t counts[256] = {0};
  std::vector<char> order;  // To preserve the first-appearance order

  for (const auto& op : operands) {
    for (char c : op) {
      if (counts[static_cast<unsigned char>(c)] == 0) {
        order.push_back(c);
      }
      counts[static_cast<unsigned char>(c)]++;
    }
  }

  indices_t result;
  for (char c : order) {
    // In standard einsum, indices appearing once are kept.
    // Indices appearing twice are contracted.
    if (counts[static_cast<unsigned char>(c)] == 1) {
      result.push_back(c);
    }
  }

  // Sort to be deterministic, as per NumPy convention
  std::sort(result.begin(), result.end());

  if (result.size() > 3) {
    throw EvaluationError("Implicit result rank > 3");
  }

  return result;
}

inline void Equation::_validate_usage(const std::vector<indices_t>& operands) {
  if (operands.size() < 2) {
    throw EvaluationError("At least two operands are required");
  }

  // 1. Map out all indices present in the input operands
  bool input_registry[256] = {false};

  for (size_t i = 0; i < operands.size() - 1; ++i) {
    for (char c : operands[i]) {
      input_registry[static_cast<unsigned char>(c)] = true;
    }
  }

  // 2. Cross-reference the result indices against the registry
  const indices_t& result_labels = operands.back();
  for (char c : result_labels) {
    if (!input_registry[static_cast<unsigned char>(c)]) {
      throw ParserError(
          0, ARMA_EINSUM_FORMAT(
                 "Result integrity violation: index '{}' appears in result but not in any input operand.", c));
    }
  }
}

inline Equation Equation::parse(const std::string& eq) {
  std::string_view equation = eq;
  if (equation.empty()) {
    throw ParserError(0, "Empty equation");
  }

  std::vector<indices_t> operands;
  indices_t current_indices;
  bool parsing_result = false;
  size_t i = 0;

  auto flush_operand = [&]() {
    if (current_indices.size() > 3) {
      throw ParserError(i, "Rank exceeds Armadillo limit (max 3)");
    }
    operands.push_back(std::move(current_indices));
    current_indices = {};
  };

  while (i < equation.length()) {
    char c = equation[i];

    if (std::isspace(c)) {
      i++;
      continue;
    }

    if (std::isalpha(c)) {
      current_indices.push_back(c);
    } else if (c == ',') {
      if (parsing_result) {
        throw ParserError(i, "Unexpected comma in result section");
      } else if (current_indices.empty()) {
        throw ParserError(i, "Empty operand (double comma?)");
      }
      flush_operand();
    } else if (c == '-') {
      if (parsing_result) {
        throw ParserError(i, "Multiple arrows detected");
      } else if (i + 1 >= equation.length() || equation[i + 1] != '>') {
        throw ParserError(i, "Expected '->'");
      }

      flush_operand();
      parsing_result = true;
      i++;  // Skip the '>'
    } else {
      throw ParserError(i, "Invalid character in equation");
    }
    i++;
  }

  // Handle the final indices (either the result or the last operand)
  flush_operand();

  // If no arrow was provided, calculate implicit labels
  if (!parsing_result) {
    operands.push_back(_calculate_implicit(operands));
  }

  return Equation(operands);
}

template <ArmadilloType... Types>
multival_t Equation::indices_size(const Types&... operands) const {
  if (sizeof...(Types) != _eq.size() - 1) {
    throw EvaluationError(ARMA_EINSUM_FORMAT("Expected {} operands, got {}", _eq.size() - 1, sizeof...(Types)));
  }

  multival_t indices_size;

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
      } else if (op.n_rows == 1 || op.n_cols == 1) {
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

  return indices_size;
}

template <typename T>
T Equation::estimate_flop_count(const multival_t& indices_size) {
  auto uniques = unique_indices();

  T flop_count = std::accumulate(uniques.begin(), uniques.end(), 1, [&](T a, auto& b) {
    return a * static_cast<T>(indices_size.at(b));
  });

  return flop_count * static_cast<T>(_eq.size() - 1);
}

template <typename T, ArmadilloType... Types>
arma::Mat<T> Equation::evaluate_mat(const Types&... operands) const {
  // 1. Basic validation of result dimensionality
  const auto& result_indices = _eq.back();
  if (result_indices.size() > 2) {
    throw EvaluationError("Result rank > 2; use evaluate_cube() instead.");
  }

  // 2. Define and check sizes
  multival_t indices_size = this->indices_size(operands...);

  // 3. Initialize result matrix
  arma::Mat<T> result;
  if (result_indices.empty()) {
    result.set_size(1, 1);
  } else if (result_indices.size() == 1) {
    result.set_size(indices_size.at(result_indices[0]), 1);
  } else {
    result.set_size(indices_size.at(result_indices[0]), indices_size.at(result_indices[1]));
  }
  result.zeros();

  // 4. Setup: Create the master label list (schema) and the ID maps
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
      } else if (op.n_rows == 1 || op.n_cols == 1) {
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

using step_t = std::array<uint64_t, 2>;

using path_t = std::vector<step_t>;

/// Method to find optimal path
enum Optimization {
  None,
  Greedy,
};

template <typename T>
class ContractionEngine {
 private:
  using RowRef = std::reference_wrapper<const arma::Row<T>>;
  using ColRef = std::reference_wrapper<const arma::Col<T>>;
  using MatRef = std::reference_wrapper<const arma::Mat<T>>;
  using CubeRef = std::reference_wrapper<const arma::Cube<T>>;

  struct Operand {
    /// Variant to hold intermediates (Mat or Cube, or reference to it)
    std::variant<arma::Mat<T>, arma::Cube<T>, ColRef, RowRef, MatRef, CubeRef> data;
    /// Associated labels
    indices_t labels;
  };

  static const arma::Mat<T>& _as_mat(const Operand& op) {
    if (std::holds_alternative<MatRef>(op.data)) {
      return std::get<MatRef>(op.data).get();
    } else if (std::holds_alternative<RowRef>(op.data)) {
      return std::get<RowRef>(op.data).get();
    } else if (std::holds_alternative<ColRef>(op.data)) {
      return std::get<ColRef>(op.data).get();
    } else {
      return std::get<arma::Mat<T>>(op.data);
    }
  }

  static const arma::Cube<T>& _as_cube(const Operand& op) {
    if (std::holds_alternative<CubeRef>(op.data)) {
      return std::get<CubeRef>(op.data).get();
    } else {
      return std::get<arma::Cube<T>>(op.data);
    }
  }

  /// Given `step`, compute the resulting equation
  static Equation _simplify(const Equation& eq, const step_t& step, const indices_t& result);

  /// Find remaining intermediates after doing the contraction at positions `a` and `b`
  static indices_t _remaining_intermediate(const Equation& eq, const step_t& step);

  /// Evaluate contraction between a pair of operands, using `transformation`.
  /// Attempt to use Armadillo function when possible, fall back to `transformation.evaluate_mat` if not.
  static Operand _evaluate_pair(const Operand& a, const Operand& b, const Equation& transformation);

  static arma::Mat<T> _evaluate_final(const Operand& a, const Equation& transformation);

 public:
  ContractionEngine() = default;

  /// Find a path to evaluate `eq`, using the "greedy" algorithm (use best contraction at each step)
  static path_t find_path_greedy(const Equation& eq, const multival_t& indices_size);

  /**
   * Evaluate `eq` by optimizing it when possible
   *
   * @tparam T a floating-point type
   * @tparam Types Armadillo objects
   * @param level level to find optimal path (may generate intermediates)
   * @param eq an equation
   * @param operands Armadillo objects
   * @return a matrix with the evaluated result
   */
  template <ArmadilloType... Types>
  arma::Mat<T> evaluate_mat(const Optimization& level, const Equation& eq, const Types&... operands) const;
};

template <typename T>
Equation ContractionEngine<T>::_simplify(const Equation& eq, const step_t& step, const indices_t& result) {
  std::vector<indices_t> operands;
  operands.assign(eq.operands().begin(), eq.operands().end());

  operands.erase(operands.begin() + static_cast<int64_t>(step[0]));
  operands.at(step[1] - 1) = result;

  return Equation(operands);
}

template <typename T>
indices_t ContractionEngine<T>::_remaining_intermediate(const Equation& eq, const step_t& step) {
  // check which indices are required (they cannot be removed)
  std::set<char> required;
  for (const auto& c : eq.operands().back()) {
    required.insert(c);
  }

  for (uint64_t iop = 0; iop < eq.n() - 1; ++iop) {
    if (iop != step[0] && iop != step[1]) {
      for (const auto& c : eq.at(iop)) {
        required.insert(c);
      }
    }
  }

  // Build the intermediate set
  indices_t intermediate;
  std::set<char> seen;  // To avoid duplicates (e.g., Hadamard indices)

  auto add_if_required = [&](const indices_t& indices) {
    for (char c : indices) {
      if (required.contains(c) && !seen.contains(c)) {
        intermediate.push_back(c);
        seen.insert(c);
      }
    }
  };

  add_if_required(eq.at(step[0]));
  add_if_required(eq.at(step[1]));

  return intermediate;
}

template <typename T>
typename ContractionEngine<T>::Operand ContractionEngine<T>::_evaluate_pair(
const Operand& a, const Operand& b, const Equation& transformation) {
  assert(transformation.n() == 3);

  auto iA = transformation.at(0);
  auto iB = transformation.at(1);
  auto iR = transformation.at(2);

#ifdef ARMA_EINSUM_DEBUG
  std::cout << " && " << std::string(transformation) << std::endl;
#endif

  if (iA.size() == 1 && iA == iB && iR.size() == 0) {  // i,i->
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use DOT" << std::endl;
#endif
    return {
      arma::Mat<T>(
        1, 1, arma::fill::value(arma::dot(_as_mat(a), _as_mat(b)))),
      iR};
  } else if (
    iA.size() == 1 && iB.size() == 1 && iR.size() == 2
    && iA.at(0) == iR.at(0) && iB.at(0) == iR.at(1)) {  // i,j-> ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use outer" << std::endl;
#endif
    return {_as_mat(a) * _as_mat(b).t(), iR};
  } else if (iA.size() == 2 && iB.size() == 1 && iA.at(1) == iB.at(0) && iR.at(0) == iA.at(0)) {  // ij,j->i
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use AXPY" << std::endl;
#endif
    return {_as_mat(a) * _as_mat(b), iR};
  } else if (iA.size() == 2 && iB.size() == 1 && iA.at(0) == iB.at(0) && iR.at(0) == iA.at(1)) {  // ji,j->i
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use AXPY" << std::endl;
#endif
    return {_as_mat(a).t() * _as_mat(b), iR};
  } else if (iA.size() == 1 && iB.size() == 2 && iA.at(0) == iB.at(1) && iR.at(0) == iB.at(0)) {  // j,ij->i
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use AXPY" << std::endl;
#endif
    return {_as_mat(b) * _as_mat(a), iR};
  } else if (iA.size() == 1 && iB.size() == 2 && iA.at(0) == iB.at(0) && iR.at(0) == iB.at(1)) {  // i,ij->j
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use AXPY" << std::endl;
#endif
    return {_as_mat(b).t() * _as_mat(a), iR};
  } else if (iA.size() == 2 && iA == iB && iB == iR) {  // ij,ij->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use Hadamard" << std::endl;
#endif
    return {_as_mat(a) % _as_mat(b), iR};
  } else if (iA.size() == 2 && iA == iB && iR.size() == 0) {  // ij,ij->
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use reduce" << std::endl;
#endif
    return {arma::Mat<T>(1, 1, arma::fill::value(arma::accu(_as_mat(a) % _as_mat(b)))), iR};
  } else if (
    iA.size() == 2 && iB.size() == 2
    && iA.at(1) == iB.at(0) && iR.at(0) == iA.at(0) && iR.at(1) == iB.at(1)) {  // ik,kj->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use GEMM" << std::endl;
#endif
    return {_as_mat(a) * _as_mat(b), iR};
  } else if (
    iA.size() == 2 && iB.size() == 2
    && iA.at(0) == iB.at(0) && iR.at(0) == iA.at(1) && iR.at(1) == iB.at(1)) {  // ki,kj->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use GEMM" << std::endl;
#endif
    return {_as_mat(a).t() * _as_mat(b), iR};
  } else if (
    iA.size() == 2 && iB.size() == 2
    && iA.at(0) == iB.at(0) && iR.at(0) == iA.at(1) && iR.at(1) == iB.at(1)) {  // ki,kj->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use GEMM" << std::endl;
#endif
    return {_as_mat(a).t() * _as_mat(b), iR};
  } else if (
    iA.size() == 2 && iB.size() == 2
    && iA.at(1) == iB.at(1) && iR.at(0) == iA.at(0) && iR.at(1) == iB.at(0)) {  // ik,jk->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use GEMM" << std::endl;
#endif
    return {_as_mat(a) * _as_mat(b).t(), iR};
  } else if (iA.size() <= 2 && iB.size() <= 2) {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use evaluate_mat" << std::endl;
#endif
    return {transformation.evaluate_mat<T>(_as_mat(a), _as_mat(b)), iR};
  } else if (iA.size() <= 2 && iB.size() == 3) {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use evaluate_mat" << std::endl;
#endif
    return {transformation.evaluate_mat<T>(_as_mat(a), _as_cube(b)), iR};
  } else if (iA.size() == 3 && iB.size() <= 2) {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use evaluate_mat" << std::endl;
#endif
    return {transformation.evaluate_mat<T>(_as_cube(a), _as_mat(b)), iR};
  } else if (iA.size() == 3 && iB.size() == 3) {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use evaluate_mat" << std::endl;
#endif
    return {transformation.evaluate_mat<T>(_as_cube(a), _as_cube(b)), iR};
  }

  throw EvaluationError("unhandled case, please report bug!");
}

template <typename T>
arma::Mat<T> ContractionEngine<T>::_evaluate_final(
const Operand& a, const Equation& transformation) {
  assert(transformation.n() == 2);

  auto iA = transformation.at(0);
  auto iR = transformation.at(1);

#ifdef ARMA_EINSUM_DEBUG
  std::cout << std::string(transformation) << std::endl;
#endif

  if (iA.size() == 2 && iR.size() == 0 && iA.at(0) == iA.at(1)) {  // ii->
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use trace" << std::endl;
#endif
    return arma::Mat<T>(1, 1, arma::fill::value(arma::trace(_as_mat(a))));
  } else if (iA.size() > 0 && iR.size() == 0) {  // i-> or ij->
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use contraction" << std::endl;
#endif
    return arma::Mat<T>(1, 1, arma::fill::value(arma::accu(_as_mat(a))));
  } else if (iA.size() == 2 && iR.size() == 2 && iA.at(0) == iR.at(1) && iA.at(1) == iR.at(0)) {  // ij->ji
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use transpose" << std::endl;
#endif
    return arma::Mat<T>(_as_mat(a).t());
  } else if (iA.size() <= 2 && iA == iR) {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use direct" << std::endl;
#endif
    return _as_mat(a);
  } else if (iA.size() <= 2) {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use evaluate_mat" << std::endl;
#endif
    return transformation.evaluate_mat<T>(_as_mat(a));
  } else if (iA.size() == 3) {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use evaluate_mat" << std::endl;
#endif
    return transformation.evaluate_mat<T>(_as_cube(a));
  }

  throw EvaluationError("unhandled case, please report bug!");
}

template <typename T>
path_t ContractionEngine<T>::find_path_greedy(const Equation& eq, const multival_t& indices_size) {
  auto ceq = eq;

  path_t result;

  while (ceq.n() > 2) {
    T cost = ceq.estimate_flop_count<T>(indices_size);
    step_t best_step = {0, 1};

    // find best step
    for (uint64_t jop = 1; jop < ceq.n() - 1; ++jop) {
      for (uint64_t iop = 0; iop < jop; ++iop) {
        if (jop == iop) {
          continue;
        }

        auto intermediates = _remaining_intermediate(ceq, {iop, jop});
        auto transformation = Equation({ceq.at(iop), ceq.at(jop), intermediates});
        auto simplified = _simplify(ceq, {iop, jop}, transformation.operands().back());

        T new_cost =
          simplified.estimate_flop_count<T>(indices_size) + transformation.estimate_flop_count<T>(indices_size);

        if (new_cost < cost) {
          cost = new_cost;
          best_step = {iop, jop};
        }
      }
    }

    auto intermediates = _remaining_intermediate(ceq, best_step);
    auto transformation = Equation({ceq.at(best_step[0]), ceq.at(best_step[1]), intermediates});
    ceq = _simplify(ceq, best_step, transformation.operands().back());

    result.push_back(best_step);
  }

  return result;
}

template <typename T>
template <ArmadilloType... Types>
arma::Mat<T> ContractionEngine<T>::evaluate_mat(
const Optimization& level, const Equation& eq, const Types&... operands) const {
  if (level != None) {
    multival_t indices_size = eq.indices_size(operands...);

    // 1. Direct Unpack into std::list
    std::list<Operand> stack;
    size_t op_idx = 0;
    ([&](const auto& op) {
      Operand opx = {std::cref(op), eq.operands()[op_idx++]};
      stack.push_back(opx);
    }(operands), ...);

    // 2. Contraction if needed
    Equation current_eq = eq;
    if (current_eq.n() > 2) {
      path_t path = find_path_greedy(current_eq, indices_size);

      for (const auto& step : path) {
        auto itA = std::next(stack.begin(), static_cast<ssize_t>(step[0]));
        auto itB = std::next(stack.begin(), static_cast<ssize_t>(step[1]));

        // Calculate intermediate labels
        indices_t inter = _remaining_intermediate(current_eq, step);

        // Construct & evaluate the transformation equation for this pair
        Equation transformation({itA->labels, itB->labels, inter});
#ifdef ARMA_EINSUM_DEBUG
        std::cout << std::string(current_eq);
#endif
        Operand result = _evaluate_pair(*itA, *itB, transformation);

        // update
        stack.erase(itA);
        *itB = std::move(result);
        current_eq = _simplify(current_eq, step, inter);
      }
    }

    return _evaluate_final(stack.front(), current_eq);
  } else {
    return eq.evaluate_mat<T>(operands...);
  }
}

/**
 * Evaluate an Einstein summation
 *
 * @warning result must be representable as a `arma::Mat<T>`
 *
 * @tparam T floating point type
 * @tparam Types Armadillo types
 * @param equation an equation
 * @param operands Armadillo objects
 * @return Result, as a matrix
 */
template <typename T, ArmadilloType... Types>
arma::Mat<T> einsum_mat(const std::string& equation, const Types&... operands) {
  return Equation::parse(equation).evaluate_mat<T>(operands...);
}

/**
 * Evaluate an Einstein summation, using an optimized path
 *
 * @warning result must be representable as a `arma::Mat<T>`
 *
 * @tparam T floating point type
 * @tparam Types Armadillo types
 * @param level level of optimization for the path
 * @param equation an equation
 * @param operands Armadillo objects
 * @return Result, as a matrix
 */
template <typename T, ArmadilloType... Types>
arma::Mat<T> einsum_mat_opt(const Optimization& level, const std::string& equation, const Types&... operands) {
  return ContractionEngine<T>().evaluate_mat(level, Equation::parse(equation), operands...);
}

}  // namespace armaeinsum


#endif  // ARMA_EINSUM_HPP_
