# `arma_einsum`

A (not clever) implementation of [`einsum()`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) in C++ using Armadillo.

## Install & use

Just copy [`arma_einsum.hpp`](arma_einsum.hpp) in your project and make sure you are also using Armadillo, and that is it:

```cpp
#define ARMA_USE_OPENMP
#include <armadillo>

#include "arma_einsum.hpp"

// have fun!
```