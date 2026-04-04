# `arma-einsum`

A lightweight (and [limited](#notes--limitations)) implementation of NumPy’s [`einsum()`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) for C++, built on top of Armadillo.

It enables expressive tensor-style operations (dot products, traces, contractions, etc.) using a compact Einstein summation notation.

## Notes & Limitations

- Requires C++20 and the support of [`std::format`](https://github.com/paulkazusek/std_format_cheatsheet/blob/main/README.md#requirement) (i.e., GCC >= 13.1 or clang >= 14)
- Supports OMP.
- Due to the use of Armadillo, only supports outputs up to **rank-2 (matrices)**.
- No support for:
    - Ellipsis (`...`)
    - Broadcasting (NumPy-style)
- No [optimization](https://optimized-einsum.readthedocs.io/en/stable/) via the creation of intermediate, so performances are suboptimal.

## Installation

Just copy [`arma_einsum.hpp`](arma_einsum.hpp) into your project and make sure Armadillo is properly set up, with OpenMP enabled.

```cpp
#define ARMA_USE_OPENMP
#include <armadillo>

#include "arma_einsum.hpp"
```

That’s it: no build system changes or additional dependencies required.

If you want to use a Meson wrap file instead, check [the example](example-meson).

## Usage

```cpp
armaeinsum::einsum_mat<T>("equation", operand1, operand2, ...);
```

### Parameters

- **`T`**: Any floating point type supported by Armadillo (mixing types is not supported).
- **`equation`**: A string describing the Einstein summation (see below).
- **`operands`**: A variadic list of Armadillo objects:
    - `arma::Col`
    - `arma::Row`
    - `arma::Mat`
    - `arma::Cube`

### Equation Format

The equation string follows standard Einstein summation notation:

```
"indices[,indices...] [-> output_indices]"
```

### Rules

- Input operands are separated by commas:
  ```
  "ij,jk"
  ```
- Repeated indices are **summed over** (contraction).
- Non-repeated indices define the output dimensions (in order of appearance).
- You can explicitly define the output using `->`:
  ```
  "ij,jk->ik"
  ```
- If `->` is omitted, output indices are inferred automatically.
- **The output must be representable as a `arma::Mat<T>`**.

## Examples

Assume:

- `a`, `b` are vectors (`arma::Col` or `arma::Row`)
- `A`, `B` are matrices (`arma::Mat`)

| Expression            | Description                   |
|-----------------------|-------------------------------|
| `("i,i", a, b)`       | Dot product                   |
| `("i,j->ij", a, b)`   | Outer product                 |
| `("ii", A)`           | Trace                         |
| `("ij->ji", A)`       | Transpose                     |
| `("ij->", A)`         | Sum of all elements           |
| `("ij,j", A, a)`      | Matrix-vector multiplication  |
| `("ik,kj", A, B)`     | Matrix-matrix multiplication  |


## Development

[Issues](https://github.com/pierre-24/arma-einsum/issues) and [pull requests](https://github.com/pierre-24/arma-einsum/pull/) are welcomed :)

To help, you can start by [forking the repository](https://github.com/pierre-24/arma-einsum/fork), and then

```bash
git clone git@github.com:YOUR_USERNAME/arma-einsum.git
cd arma-einsum
```

It is recommended to use the [Meson](https://mesonbuild.com/) build system:

```bash
# setup
meson setup _build

# compile
meson compile -C _build

# run test suites
meson test -C _build
```

You can also check linting via [`cpplint`](https://github.com/cpplint/cpplint) using:

```bash
pip install cpplint  # optional: use virtualenv

make lint
```