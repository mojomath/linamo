# MatMojo <!-- omit in toc -->

A matrix and linear algebra library for Mojo.

**[Roadmap](docs/ROADMAP.md)** | **[Mojo Miji](https://mojo-lang.com/miji/)** | **[Repository on GitHub»](https://github.com/mojomath/matmojo)** | **[Discord channel»](https://discord.gg/3rGH87uZTk)**

- [Overview](#overview)
- [Goals](#goals)
- [Background](#background)
- [Install](#install)
- [Quick start](#quick-start)
  - [Create matrices](#create-matrices)
  - [Arithmetic](#arithmetic)
  - [Linear algebra](#linear-algebra)
- [Project structure](#project-structure)
- [Status](#status)
  - [Requirements](#requirements)
- [License](#license)

## Overview

MatMojo focuses on efficient **matrix operations** and provides the foundations
for **linear algebra** workflows in Mojo.

The name **MatMojo** has a double meaning: **Mat**rix + Mojo (emphasizing the
core type) and **Mat**h + Mojo (emphasizing the linear algebra focus). Both
readings capture what this library is about.

Compared to a general-purpose multi-dimensional array library, MatMojo is more
specialized and optimized for linear algebra of 2D matrices. This allows us to
keep the API small, clean, and focused, while still providing powerful
functionality for matrix computations. It is designed to be similar to
`scipy.linalg` in Python and `nalgebra` in Rust, but with a more Mojo-idiomatic
API.

If you need multi-dimensional arrays, consider the
[NuMojo package](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).

Below are some differences between **MatMojo** (this package) and **NuMojo** (a
general-purpose multi-dimensional array library):

| Feature                  | **MatMojo**                                       | **NuMojo**                                         |
| ------------------------ | ------------------------------------------------- | -------------------------------------------------- |
| **Primary goal**         | Linear algebra & matrix computation               | General-purpose ndarray / tensor computing         |
| **Supported dimensions** | 2D only (matrices)                                | Arbitrary dimensions (N-D arrays)                  |
| **Core abstraction**     | Matrix as a mathematical object                   | N-dimensional array container                      |
| **Target domain**        | BLAS / LAPACK style workflows                     | NumPy-style scientific computing                   |
| **Storage model**        | Matrix-specific storage (row/col strides)         | Generic strided N-D storage                        |
| **Static shapes**        | First-class support (compile-time sizes)          | Not a primary focus                                |
| **View semantics**       | Safe read-only + mutable views                    | General slicing & broadcasting                     |
| **Indexing model**       | Strict matrix indexing (row, col)                 | N-dimensional indexing                             |
| **Negative indexing**    | Not supported (explicit & safe)                   | Typically supported                                |
| **Broadcasting**         | Minimal / linear-algebra oriented                 | Full NumPy-style broadcasting                      |
| **Specialized kernels**  | Matmul / decompositions / solvers                 | Elementwise & tensor ops                           |
| **Performance focus**    | SIMD & BLAS-style kernels                         | Generic tensor operations                          |
| **API philosophy**       | Mathematical clarity & safety                     | Flexibility & generality                           |
| **Typical use cases**    | Solvers, decompositions, numerical linear algebra | Scientific computing, ML preprocessing, tensor ops |

## Goals

The initial goal is to support [Mojo Miji](https://mojo-lang.com/miji/) practice
content, focus on two-dimensional matrix computing, provide simple and intuitive
syntax, and apply a series of targeted optimizations. Throughout the source
code, detailed comments and explanations are provided, under the tag
`[Mojo Miji]` to help readers understand the design decisions and implementation
details.

- Keep the API small and easy to read while learning Mojo and this package.
- Provide simple and intuitive syntax for matrix creation and operations.
- Use **safe Mojo** features and avoid unsafe code as much as possible.
- Emphasize contiguous storage for 2D matrices, but also support non-contiguous
  views through strides.
- Optimize core operations like matrix multiplication which makes this package a
  better tool if you want to only use 2D matrices.

## Background

At the moment I am still building out the project scaffolding and solidifying
the core functionality. MatMojo targets **Mojo 1.0.0**; while the language is
now stable, this package's own API is still moving quickly, so
**pull requests are not accepted at this time**. If you have any suggestions,
questions, or feedback, please feel free to open an
[issue](https://github.com/mojomath/stamojo/issues), start a
[discussion](https://github.com/mojomath/stamojo/discussions), or reach out on
our [Discord channel](https://discord.gg/3rGH87uZTk). Thank you for your
understanding!

## Install

This project uses pixi for environment management.

```bash
pixi install
```

## Quick start

Run the test suite:

```bash
pixi run test
```

### Create matrices

```mojo
import matmojo as mm

fn main() raises:
    # From nested lists
    var A = mm.matrix[DType.float64](
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]]
    )
    print(A)

    # Convenience constructors
    var I = mm.eye[DType.float64](3)       # 3×3 identity
    var Z = mm.zeros[DType.float64](2, 4)  # 2×4 zeros
    var O = mm.ones[DType.float64](3, 3)   # 3×3 ones
```

### Arithmetic

```mojo
    # Element-wise operators
    var B = A + O   # addition
    var C = A * A   # Hadamard product
    var D = A @ A   # matrix multiplication

    # Scalar operations
    from matmojo.routines.math import scalar_mul
    var scaled = scalar_mul(A, 2.0)
```

### Linear algebra

```mojo
    # Transpose & trace
    var At = mm.transpose(A)
    var t  = mm.trace(A)

    # LU decomposition (PA = LU)
    var lup = mm.lu(A)
    var L   = lup[0].copy()
    var U   = lup[1].copy()
    var piv = lup[2].copy()

    # Cholesky (A = LL^T, requires SPD matrix)
    var spd = mm.matrix[DType.float64](
        [[4.0, 12.0, -16.0],
         [12.0, 37.0, -43.0],
         [-16.0, -43.0, 98.0]]
    )
    var Lc = mm.cholesky(spd)

    # QR decomposition (A = QR)
    var qr_result = mm.qr(A)
    var Q = qr_result[0].copy()
    var R = qr_result[1].copy()
```

## Project structure

```text
matmojo
├── pixi.toml
├── src/matmojo
│   ├── __init__.mojo
│   ├── types/
│   │   ├── matrix.mojo          # Dynamic Matrix (row/col-major)
│   │   ├── matrix_view.mojo     # Non-owning view with slicing
│   │   ├── static_matrix.mojo   # Compile-time sized Matrix
│   │   └── errors.mojo          # ValueError, IndexError, etc.
│   ├── routines/
│   │   ├── creation.mojo        # matrix, zeros, ones, full, eye, diag
│   │   ├── math.mojo            # add, sub, mul, div, matmul, scalar ops
│   │   └── linalg.mojo          # transpose, trace, lu, cholesky, qr, det, solve, inv, lstsq
│   ├── traits/
│   │   └── matrix_like.mojo     # MatrixLike trait
│   └── utils/
│       ├── indexing.mojo
│       └── str.mojo
└── tests/
    ├── test_all.sh
    ├── matrix/                   # Matrix creation, indexing, lifecycle, str
    ├── matrix_view/              # View slicing, view-on-view
    ├── static_matrix/            # StaticMatrix tests
    └── routines/                 # creation, linalg, math, decompositions
```

## Status

MatMojo is under active development. See the [Roadmap](docs/ROADMAP.md) for
upcoming phases (eigenvalues, statistics, norms, etc.).

### Requirements

- Mojo `>=1.0.0,<1.1.0`
- MAX `>=26.5.0,<26.6` — supplies `parallelize()`, which moved out of the Mojo
  standard library in 1.0.0

## License

Apache License 2.0. See [LICENSE](LICENSE).
