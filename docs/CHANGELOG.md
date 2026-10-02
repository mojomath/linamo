# Linamo changelog

This is a list of changes for the Linamo package.

## 20261002 (v0.2.0)

Linamo v0.2.0 updates the package to **Mojo v1.1.0** and adds complex
matrices.

### ⭐️ New in v0.2.0

1. **Complex matrices.** `Complex[d]`, with aliases `CFloat64` and `CFloat32`,
   is a matrix element type. `lu`, `det`, `solve`, `inv`, `matrix_power`,
   `cholesky`, `qr` and `lstsq` support it, and `from_string` reads `a+bi`
   literals. It is not `Comparable`, so `sort`, `min` and `max` are absent.
1. **`conj_transpose`**, as a routine and a method, through the new
   `Conjugable` trait. It is the plain transpose for real element types.

### 🛠 Changed in v0.2.0

1. Pivoting in `lu`, `det`, `solve`, `inv` and `matrix_power` ranks by
   magnitude, not by order. Results are unchanged.
1. `Matrix.__init__(nrows, ncols, row_stride, col_stride)` accepts any
   `Numeric` element type.
1. A parameterised element type keeps its parameters in the printed header,
   e.g. `Matrix[Complex[float64]]`.
1. Dependencies: Decimo `>=0.15.0,<0.16.0` and MAX `>=26.6.0,<26.7.0`.

### 💔 Breaking in v0.2.0

1. **Linamo requires Mojo v1.1.0.** Use v0.1.0 for Mojo v1.0.0.
1. `Integer` is no longer re-exported. Use `BInt` (or `BigInt`) instead.

## 20260901 (v0.1.0)

Linamo v0.1.0 is the first release: linear algebra for Mojo, specialised for
two-dimensional matrices. It targets **Mojo v1.0.0** and depends on
**Decimo v0.13.0**, whose `Numeric` and `Parsable` traits are what let a matrix
be parameterised on an element *type* rather than a `DType` — the same
operators and routines run over `Float64` and over arbitrary-precision numbers.
Everything below is new, so this entry is an inventory of the surface rather
than a diff.

### ⭐️ New in v0.1.0

**Types:**

1. **`Matrix[T]`** — an owning, dense 2-D matrix in row- or column-major
   layout. The data buffer is a `List`, not a raw pointer, so the type is
   written in safe Mojo.
1. **`MatrixView[T]`** — a non-owning window over a `Matrix`, with strides, so
   a slice, a row, a column or a transpose costs no copy. Views are
   **read-only by default**; a mutable view exists only through the
   `mutation` module, which is what keeps an accidental write from reaching
   the matrix behind someone else's view.
1. **`StaticMatrix[T, rows, cols]`** — a matrix whose shape is a compile-time
   parameter, so a shape mismatch is a compile error rather than a raise.
   `to_matrix()` crosses over to the dynamic types.
1. The **`MatrixLike`** trait and the row/column iterators are shared by all
   three.

**Element types:**

1. Any Mojo scalar (`Float64`, `Int32`, …), plus `bool_` for the masks that
   comparisons return.
1. Decimo's exact numbers — **`BInt`**, **`Decimal`** (`BigDecimal`) and
   **`Decimal128`** — are re-exported, so `la.matrix[Decimal]` needs no second
   import. Decomposition, `det`, `solve` and `inv` run over them, which makes
   exact elimination available where binary floating point cannot say `0.3`.

**Creation:** `matrix`, `smatrix`, `from_list`, `from_string`, `zeros`,
`ones`, `full`, `empty`, `eye`, `identity`, `diag`, `arange`, `linspace`, the
`*_like` forms, and `rand` / `seed`.

**Operators and math:** `+`, `-` and unary `-` are element-wise; **`*` and `@`
are both the matrix product** and `**` is repeated multiplication
(`A**-1` inverts). The element-wise product, quotient and power have no symbol
left, so they are the `mul`, `div` and `pow` methods and routines. Scalars,
reflected operands and the in-place forms are supported throughout, alongside
`matmul`, `min`, `max`, `prod`, `sum`, `cumsum`, `cumprod`, `argmin`, `argmax`,
`sort`, `argsort` and `sort_inplace`.

**Comparison and logic:** the comparison operators return a `bool_` mask;
`isclose` / `allclose` compare approximately, the `logical_*` family combines
masks, and `all` / `any` reduce one to a verdict. Each has a `scalar_*` form
for a matrix against a single value.

**Shape, layout and mutation:** `reshape`, `reshape_view`, `resize`,
`flatten`, `contiguous`, `reorder_layout`, `broadcast_to` and `astype`. Writes
go through the `mutation` module — `view_mut`, `rows_mut`, `cols_mut`, `fill`,
`assign`, `store` — which is the library's only source of a mutable view.

**Linear algebra:** `transpose`, `trace`, `lu` (PA = LU), `cholesky`, `qr`,
`det`, `solve`, `inv`, `matrix_power` and `lstsq`. `matmul` dispatches over
four SIMD paths according to the contiguity of its operands.

**Custom operations:** `fold` and `apply_along_axis` express a reduction or a
per-row transform that the library does not name itself.

**Interoperability and printing:** `from_numpy` / `to_numpy` round-trip
through NumPy, and every type prints through one aligned grid that shows the
element type, the shape, and the strides when they are not the dense ones.
Large matrices elide rows and columns; long fractions are trimmed and marked.

**Errors:** `linamo.errors` re-exports the six kinds Linamo raises
(`ConversionError`, `IndexError`, `KeyError`, `OverflowError`, `ValueError`,
`ZeroDivisionError`) from `decimo.errors`, keeping `call_location()` pointing
at the Linamo line that raised.

**Documentation, tests and install:**

1. The **[User Manual](MANUAL.md)** is the prose tour — the two types and
   their mutability model first, since that is the part NumPy does not
   prepare you for. The per-symbol reference is in the docstrings and is
   generated with `mojo doc`.
1. **572 tests** across 37 files run under `-D ASSERT=all`, including
   differential tests against NumPy, and four runnable examples cover the
   public API of each type.
1. **`pixi add linamo`** from the
   [modular-community](https://prefix.dev/channels/modular-community/packages/linamo)
   channel brings in Mojo, MAX and Decimo. From a checkout,
   `pixi run test`, `examples` and `pack` need no separate setup step.
