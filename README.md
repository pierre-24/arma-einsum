# `arma-einsum`

A lightweight (and intentionally simple) implementation of NumPy’s [`einsum()`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) for C++, built on top of Armadillo.

It enables expressive tensor-style operations (dot products, traces, contractions, etc.) using a compact Einstein summation notation.

## Notes & Limitations

- Only supports outputs up to **rank-2 (matrices)**.
- No support for:
    - Ellipsis (`...`)
    - Broadcasting (NumPy-style)
- **No OMP acceleration at the moment** (or any other optimization, for that matter), so the performances are NOT optimal.

## Installation

Just copy [`arma_einsum.hpp`](arma_einsum.hpp) into your project and make sure Armadillo is properly set up.

```cpp
#define ARMA_USE_OPENMP
#include <armadillo>

#include "arma_einsum.hpp"
```

That’s it: no build system changes or additional dependencies required.

## Usage

```cpp
armaeinsum::einsum_mat("equation", operand1, operand2, ...);
```

### Parameters

- **`equation`**: A string describing the Einstein summation.
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

| Expression            | Description                  |
|----------------------|------------------------------|
| `("i,i", a, b)`      | Dot product                  |
| `("i,j->ij", a, b)`  | Outer product                |
| `("ii", A)`          | Trace                        |
| `("ij->ji", A)`      | Transpose                    |
| `("ij->", A)`        | Sum of all elements          |
| `("ij,j", A, a)`     | Matrix-vector multiplication |
| `("ik,kj", A, B)`    | Matrix-matrix multiplication |
