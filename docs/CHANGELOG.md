# Linamo changelog

This is a list of changes for the Linamo package.

## Unreleased

Complex matrices. `Matrix[CFloat64]` is an ordinary matrix: the operators, the
creation routines, the reductions and the decompositions keep their names.

### ⭐️ New

**Element type:**

1. **`Complex[d: DType]`**, with aliases **`ComplexFloat64`** / **`CFloat64`**
   and **`ComplexFloat32`** / **`CFloat32`** — named after the components, as
   `Float64` is. It wraps `std.complex.ComplexSIMD` and forwards every
   operator to it; what it adds is conformance to `decimo.Numeric`, which the
   stdlib type could not declare for a trait written after it. That one line
   is what opens the whole `Numeric` tier to complex matrices.
1. Also conforms to **`Rootable`** (`sqrt`, the principal root),
   **`Parsable`** (see below) and the new **`Conjugable`**. Elements carry
   `re`, `im`, `conj`, `norm`, `squared_norm` and `std`, the last being the way
   back out to `std.complex`.
1. Deliberately **not** `Comparable`: there is no ordering on the complex
   plane, so `sort`, `min`, `max` and the comparison operators are absent and
   will stay absent. Sort by a real key, usually `norm()`.

**Literals:** `from_string[CFloat64]("[[1+2i, 3-1i], [2i, -i]]")`. The syntax
is `a+bi`, which is also what a matrix prints, so the printed form reads back
in. Either part may be omitted where implied. A literal contains no spaces,
because a matrix literal is tokenized on whitespace as well as on commas.

**`conj_transpose`:** a routine and a method on `Matrix` and `MatrixView`.
`A @ conj_transpose(A)` is Hermitian, which is what `cholesky` and `qr` rest
on; the plain transpose cannot stand in. Conjugation is the identity on a real
number, so it is offered for every element type and is the plain transpose for
all but `Complex`. Dispatch goes through the new **`Conjugable`** trait, which
keeps the two overloads disjoint by construction rather than by accident.

**Linear algebra:** `lu`, `det`, `solve`, `inv` and `matrix_power` reach a
complex matrix, and so do `cholesky`, `qr` and `lstsq` — the three that the
arbitrary-precision element types cannot reach, since those have no square
root. `A ** n` works on both `Matrix` and `MatrixView`.

### 🛠 Changed

1. **Pivoting ranks by magnitude, not by order.** `lu`, `det`, `solve`, `inv`
   and `matrix_power` used to carry `Numeric & Comparable`, and spelled the
   pivot's magnitude as `-x if x < zero else x` — an `abs` wearing an order's
   clothes. The ranking is now a parameter, and the five routines share one
   body across all three element kinds. The public signatures of the existing
   overloads are unchanged, and the exact answers over `BInt`, `BDec` and
   `Decimal128` are unchanged.
1. **`Matrix.__init__(nrows, ncols, row_stride, col_stride)`** accepts any
   `Numeric` element, not only a hardware scalar. `Matrix[BInt](2, 2, 2, 1)`
   used to fail while `zeros[BInt](2, 2)` succeeded.
1. **A parameterised element type keeps its parameters in the header.**
   `Matrix[Complex[float64]]` rather than `Matrix[Complex]`, which printed the
   same for a 32-bit and a 64-bit complex matrix. Scalar, `BigInt` and
   `Decimal128` headers are unchanged.
1. **Decimo v0.14.0 is now the floor** (`>=0.14.0,<0.15.0`, up from
   `>=0.13.0,<0.14`). The bound is exclusive at both ends on purpose: 0.14.0
   retyped `Parsable.from_string` from `String` to `StringSlice`, and
   `Complex.from_string` is written against the new spelling, so the element
   types cannot conform against 0.13 and 0.14 at once. The git fallback in
   `tools/ensure_decimo.sh` moves to the v0.14.0 tag to match.

### 💔 Breaking

1. **`Integer` is no longer re-exported.** Decimo v0.14.0 removed the alias in
   favour of `BInt` / `BigInt`, both of which Linamo still re-exports, so
   `la.Integer` becomes `la.BInt` (or `la.BigInt`). It was an alias for the
   same type, so this is a rename at the call site and nothing more — no
   matrix, no routine and no result changes.

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
