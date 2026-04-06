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

  /// Parse the indices starting at `position` in `input`
  static indices_t _parse_indices(const std::string& input, uint64_t& position);

 public:
  Equation() = default;
  Equation(const Equation& o) = default;
  Equation(Equation&&) = default;
  Equation& operator=(const Equation&) = default;

  explicit Equation(const std::vector<indices_t>& indices): _eq(indices) {
    if (indices.size() < 2) {
      throw EvaluationError("not enough operands");
    }
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

/// Parse indices
indices_t Equation::_parse_indices(const std::string& input, uint64_t& position) {
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

Equation Equation::parse(const std::string& equation) {
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
  Greedy,
};

template <typename T>
class ContractionEngine {
 private:
  struct Operand {
    /// Variant to hold intermediates (Mat or Cube)
    std::variant<arma::Mat<T>, arma::Cube<T>> data;
    /// Associated labels
    indices_t labels;
  };

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
  static path_t find_path_greedy(const Equation& eq);

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
        1, 1, arma::fill::value(arma::dot(std::get<arma::Mat<T>>(a.data), std::get<arma::Mat<T>>(b.data)))),
      iR};
  } else if (
    iA.size() == 1 && iB.size() == 1 && iR.size() == 2
    && iA.at(0) == iR.at(0) && iB.at(0) == iR.at(1)) {  // i,j-> ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use outer" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(a.data) * std::get<arma::Mat<T>>(b.data).t(), iR};
  } else if (iA.size() == 2 && iB.size() == 1 && iA.at(1) == iB.at(0) && iR.at(0) == iA.at(0)) {  // ij,j->i
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use AXPY" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(a.data) * std::get<arma::Mat<T>>(b.data), iR};
  } else if (iA.size() == 2 && iB.size() == 1 && iA.at(0) == iB.at(0) && iR.at(0) == iA.at(1)) {  // ji,j->i
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use AXPY" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(a.data).t() * std::get<arma::Mat<T>>(b.data), iR};
  } else if (iA.size() == 1 && iB.size() == 2 && iA.at(0) == iB.at(1) && iR.at(0) == iB.at(0)) {  // j,ij->i
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use AXPY" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(b.data) * std::get<arma::Mat<T>>(a.data), iR};
  } else if (iA.size() == 1 && iB.size() == 2 && iA.at(0) == iB.at(0) && iR.at(0) == iB.at(1)) {  // i,ij->j
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use AXPY" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(b.data).t() * std::get<arma::Mat<T>>(a.data), iR};
  } else if (iA.size() == 2 && iA == iB && iB == iR) {  // ij,ij->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use Hadamard" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(a.data) % std::get<arma::Mat<T>>(b.data), iR};
  } else if (
    iA.size() == 2 && iB.size() == 2
    && iA.at(1) == iB.at(0) && iR.at(0) == iA.at(0) && iR.at(1) == iB.at(1)) {  // ik,kj->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use GEMM" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(a.data) * std::get<arma::Mat<T>>(b.data), iR};
  } else if (
    iA.size() == 2 && iB.size() == 2
    && iA.at(0) == iB.at(0) && iR.at(0) == iA.at(1) && iR.at(1) == iB.at(1)) {  // ki,kj->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use GEMM" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(a.data).t() * std::get<arma::Mat<T>>(b.data), iR};
  } else if (
    iA.size() == 2 && iB.size() == 2
    && iA.at(0) == iB.at(0) && iR.at(0) == iA.at(1) && iR.at(1) == iB.at(1)) {  // ki,kj->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use GEMM" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(a.data).t() * std::get<arma::Mat<T>>(b.data), iR};
  } else if (
    iA.size() == 2 && iB.size() == 2
    && iA.at(1) == iB.at(1) && iR.at(0) == iA.at(0) && iR.at(1) == iB.at(0)) {  // ik,jk->ij
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use GEMM" << std::endl;
#endif
    return {std::get<arma::Mat<T>>(a.data) * std::get<arma::Mat<T>>(b.data).t(), iR};
  } else {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use evaluate_mat" << std::endl;
#endif
    return {transformation.evaluate_mat<T>(std::get<arma::Mat<T>>(a.data), std::get<arma::Mat<T>>(b.data)), iR};
  }
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
    return arma::Mat<T>(1, 1, arma::fill::value(arma::trace(std::get<arma::Mat<T>>(a.data))));
  } else if (iA.size() > 0 && iR.size() == 0) {  // i-> or ij->
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use contraction" << std::endl;
#endif
    return arma::Mat<T>(1, 1, arma::fill::value(arma::accu(std::get<arma::Mat<T>>(a.data))));
  } else if (iA.size() == 2 && iR.size() == 2 && iA.at(0) == iR.at(1) && iA.at(1) == iR.at(0)) {  // ij->ji
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use transpose" << std::endl;
#endif
    return arma::Mat<T>(std::get<arma::Mat<T>>(a.data).t());
  } else {
#ifdef ARMA_EINSUM_DEBUG
    std::cout << "* use direct" << std::endl;
#endif
    return std::get<arma::Mat<T>>(a.data);
  }
}

template <typename T>
path_t ContractionEngine<T>::find_path_greedy(const Equation& eq) {
  auto ceq = eq;

  path_t result;

  while (ceq.n() > 2) {
    uint64_t cost = ceq.length();
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

        if (simplified.length() < cost) {
          cost = simplified.length();
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
template <ArmadilloType ... Types>
arma::Mat<T> ContractionEngine<T>::evaluate_mat(
const Optimization& level, const Equation& eq, const Types&... operands) const {
  // Unpack variadic operands into a stack using an index sequence
  std::vector<Operand> stack;
  stack.reserve(sizeof...(Types));
  auto unpack = [&]<std::size_t... I>(std::index_sequence<I...>) {
    ([&](auto idx, const auto& op) {
        stack.push_back({op, eq.operands()[idx]});
    }(I, operands), ...);
  };  // NOLINT

  unpack(std::make_index_sequence<sizeof...(Types)>{});

  // use a sequence of transformations
  auto ceq = eq;
  if (ceq.n() > 2) {
    for (auto& step : find_path_greedy(eq)) {
#ifdef ARMA_EINSUM_DEBUG
      std::cout << std::string(ceq);
#endif
      auto intermediates = _remaining_intermediate(ceq, step);
      auto transformation = Equation({ceq.at(step[0]), ceq.at(step[1]), intermediates});

      auto opA = std::move(stack[step[0]]);
      auto opB = std::move(stack[step[1]]);

      stack.erase(stack.begin() + static_cast<int64_t>(step[0]));

      stack[step[1] - 1] = _evaluate_pair(opA, opB, transformation);

      ceq = _simplify(ceq, step, transformation.operands().back());
    }
  }

  // final transformation to get the result
  return _evaluate_final(stack[0], ceq);
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
