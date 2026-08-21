# Linamo User Manual

A working guide to the library: what the types are, what you can do to them,
and why a few things are spelled the way they are.

This manual is written for someone who knows NumPy and is meeting Mojo's
ownership rules for the first time. Most of Linamo can be guessed at from
NumPy. The part that cannot is [The two types](#the-two-types), which is
therefore where the manual starts — read that chapter and the rest will not
surprise you.

The per-symbol reference (every signature, parameter and raise) lives in the
docstrings and is generated with `mojo doc`; see
[Generating the symbol reference](#generating-the-symbol-reference). This
manual is the prose half: the shape of the API, not an enumeration of it.

- [Linamo User Manual](#linamo-user-manual)
  - [Getting started](#getting-started)
    - [The element type is a type](#the-element-type-is-a-type)
    - [What goes in the brackets](#what-goes-in-the-brackets)
    - [Generating the symbol reference](#generating-the-symbol-reference)
  - [The two types](#the-two-types)
    - [Mutability of indexing and slicing](#mutability-of-indexing-and-slicing)
      - [From a `Matrix`](#from-a-matrix)
      - [From a `MatrixView`](#from-a-matrixview)
    - [Writing](#writing)
    - [Which conversions exist](#which-conversions-exist)
    - [Two things that are deliberately absent](#two-things-that-are-deliberately-absent)
    - [Why views are read-only](#why-views-are-read-only)
  - [Creating matrices](#creating-matrices)
    - [Ranges, shapes copied from another matrix, and random values](#ranges-shapes-copied-from-another-matrix-and-random-values)
    - [Parsing a matrix from text](#parsing-a-matrix-from-text)
  - [Printing a matrix](#printing-a-matrix)
    - [What a grid leaves out](#what-a-grid-leaves-out)
    - [Tuning the appearance](#tuning-the-appearance)
  - [Indexing and slicing](#indexing-and-slicing)
  - [Copying and assignment](#copying-and-assignment)
  - [Operators](#operators)
    - [Element-wise product, quotient and power](#element-wise-product-quotient-and-power)
    - [Matrix power](#matrix-power)
    - [Comparisons return masks](#comparisons-return-masks)
    - [Reflected operators](#reflected-operators)
    - [In-place operators](#in-place-operators)
  - [Mutating a matrix](#mutating-a-matrix)
  - [Iteration](#iteration)
  - [Reductions, searches and sorts](#reductions-searches-and-sorts)
  - [Custom reductions](#custom-reductions)
  - [Shape and layout](#shape-and-layout)
    - [The two layouts](#the-two-layouts)
    - [Reshaping routines](#reshaping-routines)
  - [Linear algebra](#linear-algebra)
  - [SIMD access](#simd-access)
  - [NumPy interoperability](#numpy-interoperability)
  - [NuMojo interoperability](#numojo-interoperability)
    - [Contiguity is not required](#contiguity-is-not-required)
    - [The one rule: do not reallocate](#the-one-rule-do-not-reallocate)
  - [Errors](#errors)
  - [Arbitrary-precision elements](#arbitrary-precision-elements)
    - [How it works, and what it costs](#how-it-works-and-what-it-costs)
    - [Decimo is a hard dependency](#decimo-is-a-hard-dependency)
  - [StaticMatrix](#staticmatrix)
    - [Crossing over to `Matrix`](#crossing-over-to-matrix)
  - [Appendix A: how it works inside](#appendix-a-how-it-works-inside)
    - [`Matrix` and `MatrixView` inter-operate](#matrix-and-matrixview-inter-operate)
    - [Why matmul has several implementations](#why-matmul-has-several-implementations)
  - [Appendix B: what is not here yet](#appendix-b-what-is-not-here-yet)

---

## Getting started

Linamo targets Mojo `1.0.0` and MAX `>=26.5.0`. The repository uses pixi:

```bash
pixi install
pixi run test
```

There is no published package yet, so a program outside the repository is
compiled with the source directory on the import path:

```bash
pixi run mojo run -I src my_program.mojo
```

One import gets you the library:

```mojo
import linamo as la          # la.matrix, la.zeros, la.transpose, ...
```

Everything a program reaches for is re-exported from
`src/linamo/__init__.mojo`, so `la.<name>` is the only spelling needed --- the
types, every routine, the mutating routines in `linamo.routines.mutation`, and
the arbitrary-precision element types from Decimo. Anything not there is
reached by its module path.

Scalar element types are spelled with the stdlib's own names:

```mojo
var A = la.matrix[Float64](...)
var B = la.matrix[Int32](...)
```

Linamo aliases none of them. A matrix is parameterised on an element *type*
rather than on a `DType`, so `Float64` already names what goes in the brackets
and a second spelling for it would buy nothing. The one exception is
`la.bool_`, the element of a comparison mask, which the stdlib has no name for.

Nothing is exported unqualified. `from linamo.prelude import *` gives you `la`
and nothing else --- a library that puts bare names in your global namespace
has overstepped.

A first program:

```mojo
import linamo as la

def main() raises:
    var A = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var B = la.eye[Float64](2)
    print(A @ B)
    print(la.det(A))
```

### The element type is a type

A matrix names its element type the way every other container does:
`Matrix[Float64]`, `Matrix[Int32]`, `Matrix[la.bool_]`, beside `List[Float64]`.
The parameter defaults to `Float64` almost everywhere, so `la.zeros(2, 3)` is a
float64 matrix.

`Float64` *is* `Scalar[DType.float64]`, so nothing about the scalar case is
lost by spelling it this way — the SIMD routines recover the dtype by matching
`Matrix[Scalar[d]]`, which infers `d` — and an element type with no `DType`
behind it at all becomes expressible. Where the stdlib has no name for one,
Linamo supplies it: `la.bool_` is `Scalar[DType.bool]`, which is a different
type from Mojo's `Bool`.

`StaticMatrix` is spelled the same way, `StaticMatrix[Float64, 2, 3]`, but only
ever accepts a scalar: its buffer is one SIMD register, and an element type
that has no register to live in is a compile error naming `Matrix` as the
alternative.

The default is *not* available on the routines that take a list, though —
`la.matrix([[1.0, 2.0]])` does not compile, and `la.matrix[Float64](...)`
is required. The reason is that a list literal has no type of its own until it
is given one, and here it would have to be given `List[List[T]]` with `T` still
unknown; the compiler cannot resolve either side first. Name the element type,
or give the list a type of its own first:

```mojo
var rows: List[List[Float64]] = [[1.0, 2.0], [3.0, 4.0]]
var A = la.matrix(rows)          # element type inferred from the argument
```

### What goes in the brackets

If a name takes square brackets, everything in them is yours to write. The
parameters Linamo works out for itself — the element type of an argument you
already passed, the origin a view borrows from — sit behind a `//` in the
signature and are unwritable:

```mojo
def sum[dtype: DType, origin: Origin[mut=False], //](
    m: MatrixView[Scalar[dtype], origin]
) -> Scalar[dtype]:
```

So `la.sum(m)` is the only spelling; `la.sum[DType.float64](m)` is rejected
with *unexpected parameter*. Nothing is lost, because there was never a second
answer available: `dtype` can only be whatever `m` holds.

The marker also renumbers the slots, so the first parameter you *are* meant to
write is always the first one in the brackets:

```mojo
m.load[4](0, 0)                    # 4 is the SIMD width, not the dtype
la.apply_along_axis[0, my_lane](m) # 0 is the axis
```

Read a bracket list, then, as the list of decisions the library cannot make for
you: the element type of a matrix conjured out of nothing (`la.zeros[BInt]`), a
SIMD width, an axis, a comparison kernel.

### Generating the symbol reference

```bash
pixi run mojo doc -I src src/linamo -o api.json
```

This walks every docstring in the package and emits a structured JSON reference.
Modular's tool does not yet render HTML, so the JSON is the artefact for now.

---

## The two types

| Type           | Owns its data | Mutable                       | Cost to create |
| -------------- | ------------- | ----------------------------- | -------------- |
| `Matrix`       | yes           | yes                           | allocates      |
| `MatrixView`   | no            | only from `routines.mutation` | O(1)           |
| `StaticMatrix` | yes, inline   | yes                           | no allocation  |

`Matrix` is the owner. Its elements live in one contiguous block of memory, in
either row-major (C-contiguous) or column-major (Fortran-contiguous) order, and
the matrix is responsible for that memory.

`MatrixView` is a non-owning window onto somebody else's memory. It is five
integers and a `Span`: offset, shape and a stride per axis. Views are how you
name a sub-matrix, a row, a column, or the same buffer under different strides,
and creating one copies nothing.

The view remembers *whose* memory it is looking at, in its type. That is the
`origin` parameter, and it is what stops a view from outliving the matrix
behind it — the compiler will not destroy a matrix while a view of it is still
alive. No unsafe pointer ever appears in a public signature.

Whether a view can *write* to that data is carried in the type too, as the
`mut` field of its origin. No method on either type hands out a writable view:
`m.view()`, `m[a:b, c:d]`, `rows()`, `cols()` and iteration are all read-only,
however the receiver was bound. A writable view comes only from
`linamo.routines.mutation`, and writing through a read-only one is a compile
error rather than a runtime check.

### Mutability of indexing and slicing

The rule is one sentence:

> **Nothing that carries a borrow in its type is ever handed out mutable,
> except through a function in `linamo.routines.mutation`.**

A `MatrixView` carries a borrow in its type (that is what its `origin`
parameter is) so every view produced by a method is read-only, no matter how
the receiver was bound. Writing is the rarer case and gets the explicit
spelling.

#### From a `Matrix`

A matrix `m` is **writable** when you own it --- `var m = la.zeros[...](3, 3)`
--- or when a function received it as a `mut` argument. It is **read-only**
when a function received it under the default `read` convention, or when you
reached it through a read-only reference.

In the table below, `m` is the matrix on the left of the expression:

| Expression            | You get      | Result if `m` is writable | Result if `m` is read-only | Rule     |
| --------------------- | ------------ | ------------------------- | -------------------------- | -------- |
| `m[i, j]`             | reference    | **mutable**               | read-only                  | inherits |
| `m[a:b, c:d]`         | `MatrixView` | read-only                 | read-only                  | fixed    |
| `m.view()`            | `MatrixView` | read-only                 | read-only                  | fixed    |
| `m.rows()`/`m.cols()` | views        | read-only                 | read-only                  | fixed    |
| `for r in m`          | views        | read-only                 | read-only                  | fixed    |

Read the first row as: *`m[i, j]` gives you a reference to one element; that
reference is mutable if `m` is writable, and read-only if `m` is read-only.*
The last column names the pattern --- **inherits** means the result takes the
receiver's mutability, **fixed** means the result is read-only either way.

`m[i, j]` is the one place a method still propagates the caller's mutability,
and it is the mutation you want on an owner: `m[i, j] = x` writes an element.
It is safe to inherit because a bare reference is consumed where it is formed,
unlike a view, which is a value you can bind, store and pass twice.

`m.view()` is exactly `m[:, :]`. The named routines accept a `Matrix` without
it --- the same conversion happens implicitly (see
[Appendix A](#appendix-a-how-it-works-inside)) --- so `view()` is for the times
you want to name the view, or to be explicit about where the borrow starts.

#### From a `MatrixView`

A view's own mutability is carried in its type, as the `mut` field of its
`origin` parameter - not as a runtime flag. A view is mutable only if it came
from a spelling with `_mut` in its name.

Same layout as above, with `v` the view on the left of the expression:

| Expression            | You get      | Result if `v` is mutable | Result if `v` is read-only | Rule     |
| --------------------- | ------------ | ------------------------ | -------------------------- | -------- |
| `v[i, j]`             | reference    | **mutable**              | read-only                  | inherits |
| `v[a:b, c:d]`         | `MatrixView` | read-only                | read-only                  | fixed    |
| `v.rows()`/`v.cols()` | views        | **mutable**              | read-only                  | inherits |
| `v.as_imm()`          | `MatrixView` | read-only                | read-only                  | fixed    |
| `v.to_matrix()`       | `Matrix`     | new owner                | new owner                  | copies   |

Every *fixed* row is one-way: nothing turns a read-only view back into a
mutable one. The *inherits* rows on a view are harmless precisely because a
mutable view can only have come from a `_mut` spelling in the first place.

### Writing

| You want to                | On a `Matrix`              | On a writable `MatrixView`   |
| -------------------------- | -------------------------- | ---------------------------- |
| write one element          | `m[i, j] = x`              | `v[i, j] = x`                |
| write the whole thing      | `m.set(value)`             | `fill(v, value)`             |
| copy another matrix in     | `m.set(src)`               | `assign(v, src)`             |
| fill a region              | `m.set(rows, cols, value)` | `fill(v, rows, cols, value)` |
| copy a block into a region | `m.set(rows, cols, src)`   | `assign(v, rows, cols, src)` |
| write rows in a loop       | `for row in rows_mut(m)`   | -                            |
| get a writable view        | `m.view_mut(x, y)`         | `view_mut(v, x, y)`          |

Every write on an owned matrix is spelled `set`; which one runs is decided by
what you pass. A `Self.ElementType` fills, a `Matrix` or `MatrixView` copies.
Nothing in the library converts a scalar to a matrix, so there is no ambiguity
to keep track of.

The right-hand column, and `rows_mut`, come from `linamo.routines.mutation`.
`Matrix.view_mut` is a method, so a writable view of an owned matrix needs no
import. Every spelling that produces a mutable view carries `_mut` in its name;
nothing else in the library grants write access.

### Which conversions exist

| From \ To              | `Matrix`        | `MatrixView` read-only | `MatrixView` mutable | element ref read-only | element ref mutable |
| ---------------------- | --------------- | ---------------------- | -------------------- | --------------------- | ------------------- |
| `var Matrix`           | `m.copy()`      | `m[a:b, c:d]`          | `m.view_mut(x, y)`   | -                     | `m[i, j]`           |
| read-only `Matrix`     | `m.copy()`      | `m.view()`             | **impossible**       | `m[i, j]`             | **impossible**      |
| read-only `MatrixView` | `v.to_matrix()` | `v[a:b, c:d]`          | **impossible**       | `v[i, j]`             | **impossible**      |
| mutable `MatrixView`   | `v.to_matrix()` | `v.as_imm()`           | `view_mut(v, x, y)`  | `v.as_imm()[i, j]`    | `v[i, j]`           |

The mutable column has exactly one spelling, and the two **impossible** columns
are the invariant the design rests on: mutability is only ever lost, never
gained. Nothing in the library promotes a read-only value to a writable one.

The audit is mechanical. A method can only hand back the caller's mutability by
taking `ref self`, so:

```console
$ grep -rn "ref self," src/linamo/types/
src/linamo/types/matrix.mojo:209:        ref self, row: Int, col: Int
src/linamo/types/matrix.mojo:562:        ref self, x: Slice, y: Slice
```

Two lines: element access, and `view_mut`. Both are named in the tables above.
Anything else appearing there is a hole.

### Two things that are deliberately absent

`m[a:b, c:d] = src` is not available. Defining `__setitem__` on `Matrix` makes
the compiler pass `self` to `__getitem__` as a temporary copy in some
positions, so a sliced view ends up carrying the origin of a dead temporary and
`a[0:1, :] - a[1:2, :]` stops compiling. Region assignment is `m.set(...)`.

`m.view_mut(a:b, c:d)` is not available either: `a:b` is subscript syntax, not
expression syntax, so no plain call can accept it. Pass `Slice(a, b)`.

### Why views are read-only

It is worth understanding why the rule is fixed rather than inherited, because
it is the difference between the library being usable and not.

A mutable view is an *exclusive* borrow of the matrix behind it. Mojo will not
let two values that both carry a mutable borrow of the same memory be passed to
one call - and it will not allow a mutable and a read-only borrow together
either. If views inherited mutability, then on a `var` matrix the most ordinary
expressions in linear algebra would be rejected by the compiler:

```mojo
var a = la.matrix[Float64]([[10.0, 20.0], [1.0, 2.0]])

var d = a[0:1, :] - a[1:2, :]      # two views of `a` in one call
var p = a[0:2, 0:2] @ a[0:2, 0:2]
var c = a + a[0:2, 0:2]
```

Reading the same matrix twice at once is always safe, so views are read-only
and all three lines compile.

This is the same choice Rust's `ndarray` makes - `slice()` read-only,
`slice_mut()` behind `&mut self` - and for the same reason. NumPy and Eigen
hand out mutable views from slicing, but neither has a borrow checker to
answer to; with one, "less safe" does not show up as corruption later, it shows
up as compile errors on correct-looking code.

A mutable view, once you ask for one, still cannot appear twice in one
expression. `as_imm()` demotes it, the same way `Span.as_imm()` does:

```mojo
var v = view_mut(a, Slice(0, 2), Slice(0, 2))
var c = v.as_imm() + v.as_imm()
```

---

## Creating matrices

| Call                                               | Gives you                                     |
| -------------------------------------------------- | --------------------------------------------- |
| `matrix[T](list_of_rows, order="C")`               | a matrix from nested lists                    |
| `matrix[T](flat_list=..., nrows=, ncols=, order=)` | a matrix from one flat list                   |
| `zeros[T](nrows, ncols)`                           | all zeros                                     |
| `ones[T](nrows, ncols)`                            | all ones                                      |
| `full[T](nrows, ncols, fill_value)`                | one repeated value                            |
| `eye[T](n)` / `identity[T](n)`                     | the `n x n` identity                          |
| `diag[T](values)`                                  | a square matrix with `values` on the diagonal |
| `diag[T](m)`                                       | the diagonal of `m`, as a `List`              |
| `smatrix[nrows, ncols, T](list_of_rows)`           | a `StaticMatrix`                              |
| `empty[T](nrows, ncols)`                           | uninitialised storage of that shape           |
| `zeros_like(m)` / `ones_like(m)`                   | zeros or ones shaped like `m`                 |
| `full_like(m, fill_value)` / `empty_like(m)`       | one value, or uninitialised, shaped like `m`  |
| `arange[T](stop)` / `arange[T](start, stop, step)` | a `1 x n` row of evenly spaced values         |
| `linspace[T](start, stop, num, endpoint)`          | a `1 x num` row from `start` to `stop`        |
| `from_list[T](flat_list, nrows, ncols, order)`     | a matrix from one flat list, positionally     |
| `from_string[T](text)`                             | a matrix parsed from a literal                |
| `rand[T](nrows, ncols, low, high)`                 | uniform random values                         |
| `seed(value)`                                      | pins `rand` for reproducibility               |

```mojo
import linamo as la

def main() raises:
    var A = la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = la.matrix[Float64](
        flat_list=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], nrows=2, ncols=3
    )
    var F = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]], order="F")

    var Z = la.zeros[Float64](2, 3)
    var O = la.ones[Int32](2, 3)
    var C = la.full[Float64](2, 2, 7.5)
    var I = la.eye[Float64](3)
    var D = la.diag[Float64]([1.0, 2.0, 3.0])
```

`order` chooses the memory layout: `"C"` for row-major, `"F"` for column-major.
It changes where the elements sit in memory and nothing else — `A[i, j]` means
the same thing either way. See [Shape and layout](#shape-and-layout).

The list-taking forms raise `ValueError` on an empty list, on rows of different
lengths, on a `flat_list` whose length is not `nrows * ncols`, or on an `order`
other than `"C"` or `"F"`. The shape-only routines — `zeros`, `ones`, `full`,
`eye`, `identity` — cannot fail and are not `raises` at all.

The `[T]` is required whenever a list is passed and optional otherwise; see
[The element type is a type](#the-element-type-is-a-type) for why.

### Ranges, shapes copied from another matrix, and random values

`arange` and `linspace` return a **`1 x n` row**, because Linamo has no 1-D type
and a row is what NumPy's 1-D result prints as. `reshape(x, n, 1)` gives the
column.

```mojo
var x = la.arange[Float64](5.0)              # 1x5: 0 1 2 3 4
var y = la.arange[Float64](1.0, 2.0, 0.25)   # 1x4: 1 1.25 1.5 1.75
var d = la.arange[Int64](10, 0, -3)          # 1x4: 10 7 4 1
var t = la.linspace[Float64](0.0, 1.0, 5)    # 1x5: 0 0.25 0.5 0.75 1
var h = la.linspace[Float64](0.0, 1.0, 5, endpoint=False)
```

`arange` excludes `stop`, as Python's `range` does, and `linspace` includes it
by default — and hits it *exactly*, rather than landing a rounding error short.
Both **raise rather than return an empty matrix**: `arange(5.0, 0.0)`, a zero
`step`, and `linspace(..., num=0)` are all `ValueError`, because a `1 x 0`
matrix cannot be printed, indexed or multiplied by anything.

The `*_like` family copies the shape and dtype of an existing matrix or view,
never its layout — the result is always C-contiguous, like every other owning
result. Use `astype` to change the dtype.

```mojo
var A = la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
var Z = la.zeros_like(A)          # 2x3 of zeros
var S = la.zeros_like(A[0:2, 1:3])  # 2x2 — a view works too
var E = la.empty[Float64](2, 2)  # contents unspecified; write before reading
```

`rand` draws uniformly from the closed interval `[low, high]`, defaulting to
`[0, 1]`. It uses the standard library's global generator, so `seed(n)` makes a
run reproducible:

```mojo
la.seed(42)
var R = la.rand[Float64](2, 3)         # values in [0, 1]
var Q = la.rand[Float64](3, 3, -2.0, 2.0)
var K = la.rand[Int64](2, 2, 1, 6)     # integers, both bounds included
```

### Parsing a matrix from text

`from_string` reads a literal in which elements are separated by whitespace or
commas and rows by nested brackets:

```mojo
var A = la.from_string[Float64]("[[1, 2, 3], [4, 5.5, 6]]")  # 2x3
var B = la.from_string[Float64]("[[1 2 3]\n [4 5 6]]")       # 2x3
var C = la.from_string[Float64]("1 2 3")                     # 1x3, one row
var D = la.from_string[Int32]("[1, 2, 3, 4]", 2, 2)          # shape given
```

A literal with no nesting is a single row. The second overload takes an
explicit `nrows`, `ncols` and `order`, and ignores the bracket structure
entirely — it reads every element it finds, in `order`, into the shape you
asked for. Unbalanced brackets, nesting more than two deep, rows of unequal
length, and a cell that is not a number all raise `ValueError`; the last of
these names the offending token.

`from_list` is the same thing for a list you already have — the positional
spelling of the keyword-only `matrix(flat_list=..., nrows=..., ncols=...)`.

`from_string` takes the arbitrary-precision element types too:

```mojo
var E = la.from_string[la.Dec128]("[[0.1, 0.2], [0.3, 0.4]]")
var F = la.from_string[la.BDec]("[[1.5], [2.5]]")
var G = la.from_string[la.BInt]("[[170141183460469231731687303715884105728]]")
```

For those types this is the short spelling of something you can always do by
hand — `la.matrix[la.Dec128]([[la.Dec128("0.1"), la.Dec128("0.2")]])` builds
the same matrix, one element at a time. What you cannot do is hand the routine
a list of `Float64` and let the conversion happen: `Dec128` and `BDec` have no
implicit constructor from one, deliberately, because it would round the literal
to the nearest binary float before the element ever saw it and `0.1` would
arrive as 0.1000000000000000055511151231257827. `la.Dec128(1.1)` is available
if you want it, but it goes through `from_float` and inherits Float64's limits
— decimo documents it as reliable to about 15 significant digits. Spelled from
text, `0.1` is a tenth, and `la.sum` over `E` above is exactly `1`.

`BInt` is capped the same way for a different reason: the integer in `G` is
2^127, one past what any built-in width holds, so there is no literal type it
could have passed through on the way in.

---

## Printing a matrix

`print` writes a header line and then the grid; `__str__` writes the grid on
its own, for when a matrix goes inside a larger piece of text.

```mojo
var A = la.matrix[Float64](
    [[1457.2, 9.5, 1589.62], [3.25, 1626.8, 12.5], [1648.0, 1726.0, 1804.0]]
)
print(A)
```

```console
Matrix[float64] 3x3
[[ 1457.2      9.5  1589.62 ]
 [    3.25  1626.8    12.5  ]
 [ 1648.0   1726.0  1804.0  ]]
```

Cells are padded on both sides of the decimal point, so the points of a column
stand in one vertical line. This is the alignment NumPy uses, and it is what
keeps a column readable when it holds `1648.0` next to `3.25`; flush-left or
flush-right padding lines up an edge of the number rather than its scale.
Widths are measured per column, so one wide column does not stretch the rest.

The header names the type, the element type and the shape. Strides and offset
join it only when they are not the ones a freshly built matrix has:

```console
Matrix[float64] 3x3                                  # a plain matrix
MatrixView[float64] 2x2, strides (4, 1), offset 5    # a window into one
```

A view of a whole matrix prints as cleanly as the matrix does. A view of part
of one says so, which is the case where the layout is the thing worth knowing.

### What a grid leaves out

Rows are dropped by element count and columns by line width, each marked with
`...`:

```console
Matrix[float64] 40x40
[[ 0.0  0.0  0.0  ...  0.0  0.0  0.0 ]
 [ 0.0  0.0  0.0  ...  0.0  0.0  0.0 ]
 [ 0.0  0.0  0.0  ...  0.0  0.0  0.0 ]
 ...
 [ 0.0  0.0  0.0  ...  0.0  0.0  0.0 ]
 [ 0.0  0.0  0.0  ...  0.0  0.0  0.0 ]
 [ 0.0  0.0  0.0  ...  0.0  0.0  0.0 ]]
```

The two rules are separate because a count on its own cannot protect a
terminal: an arbitrary-precision element can be wider than a line by itself, so
how many columns fit is a question about their rendered width, not their
number. Under it sits a floor — a matrix always shows `MIN_COLS_SHOWN` columns,
even where that overruns the line, because one column says less about a matrix
than a long line does.

An element with a long fractional part is trimmed rather than rounded:

```mojo
print(la.from_string[la.BDec]("[[1.0]]") / la.from_string[la.BDec]("[[3.0]]"))
```

```console
Matrix[BigDecimal] 1x1
[[ 0.33333333… ]]
```

Only digits after the decimal point are dropped, and the cut is marked. The
integer part is never touched, so a printed magnitude is always the magnitude
held: `la.from_string[la.BInt]("[[170141183460469231731687303715884105728]]")`
prints all thirty-nine of its digits. The trailing `…` is what separates an
abridged reading from a rounded value — nothing here silently reports a number
it is not holding.

### Tuning the appearance

The knobs live in `linamo/utils/formatting.mojo`:

| Alias             | Default    | Governs                                        |
| ----------------- | ---------- | ---------------------------------------------- |
| `MAX_LINE_WIDTH`  | 88         | the widest line a grid may occupy              |
| `PRINT_THRESHOLD` | 1000       | the element count above which rows elide       |
| `EDGE_ITEMS`      | 3          | rows, and columns, kept at each end            |
| `MIN_COLS_SHOWN`  | 3          | columns shown however wide they are            |
| `MAX_FRAC_DIGITS` | 8          | digits kept after the decimal point            |
| `COLUMN_GAP`      | two spaces | what separates two cells                       |
| `EDGE_PAD`        | one space  | what separates a row's brackets from its cells |
| `ELISION`         | `...`      | the stand-in for what is not shown             |
| `TRIM_MARK`       | `…`        | the mark that ends a trimmed fraction          |

They are `comptime` aliases because Mojo has no global variables yet. When it
gains them, these are what a configuration type would carry, alongside the
working precision of the arbitrary-precision element types.

All three matrix types print through this one module, so `Matrix`,
`MatrixView` and `StaticMatrix` differ in their header line and nowhere else.

---

## Indexing and slicing

```mojo
var A = la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

var x = A[1, 2]          # 6.0 — a single element
A[0, 0] = 10.0           # writes through the reference

var v = A[0:2, 1:3]      # a 2 x 2 MatrixView, nothing copied
var w = A[:, 0:1]        # the first column, as a 2 x 1 view
var s = A[0:2:2, 0:3:2]  # strided: every other row, every other column
```

Indexing with two integers yields a reference to the element, so it reads on
the right of `=` and writes on the left. Indexing with two slices yields a
`MatrixView`, always read-only.

Both indices are checked, and an out-of-range index raises `IndexError`.
Negative indices are **not** supported — `A[-1, 0]` is an error, not the last
row. This is deliberate: in a library where an index may also be a stride
offset, silent wraparound hides bugs rather than saving keystrokes.

Slices behave like Python's: `start:stop`, `start:stop:step`, and any part may
be omitted. A view of a view slices relative to the view:

```mojo
var v = A[0:2, 0:3]
var inner = v[0:1, 1:3]   # relative to v, not to A
```

Views are cheap and compose freely, and a strided view is never a special case
a routine has to be told about — every routine in the library accepts one.

To turn a view back into an owning matrix, call `to_matrix()`. It always
produces dense, C-contiguous storage, however strided the source was.

---

## Copying and assignment

`Matrix` is explicitly copyable but not *implicitly* copyable, so a plain
`var a = b` will not silently duplicate a buffer - the compiler rejects it and
asks you to say what you meant. `MatrixView` is implicitly copyable, because
copying one only duplicates a span and five integers.

| Expression              | Result                            | Cost   |
| ----------------------- | --------------------------------- | ------ |
| `var a = b`             | compile error, suggests `.copy()` | -      |
| `var a = b.copy()`      | `Matrix`, deep copy               | O(n*m) |
| `var a = b^`            | `Matrix`, moved                   | O(1)   |
| `var v = b.view()`      | `MatrixView`                      | O(1)   |
| `var v = b[a:b, c:d]`   | `MatrixView`                      | O(1)   |
| `var v2 = v`            | `MatrixView`, handle copy         | O(1)   |
| `var a = v.to_matrix()` | `Matrix`, materialised            | O(n*m) |

`copy()` on a view returns another *view* of the same data, which is why
materialising a view into owned storage has its own name, `to_matrix()`. It is
the one place in the view API that allocates, and it always produces dense,
C-contiguous storage regardless of how strided the source was.

---

## Operators

Both `Matrix` and `MatrixView` carry the full arithmetic operator set. Every
binary operator accepts a `Matrix`, a `MatrixView`, or a scalar on the right,
and always returns a **new** `Matrix` that owns its data - an operator never
writes into an operand.

| Operator                    | Meaning                                         |
| --------------------------- | ----------------------------------------------- |
| `*` `@`                     | Matrix multiplication - two names, one product  |
| `**`                        | Matrix power: `A ** 3` is `A @ A @ A`           |
| `+` `-`                     | Element-wise arithmetic                         |
| `//` `%`                    | Element-wise floor division and modulo          |
| `<` `<=` `>` `>=` `==` `!=` | Element-wise mask, `Matrix[Scalar[DType.bool]]` |

**`*` between two matrices is the matrix product, not the element-wise one.**
This is the point where linamo parts company with NumPy, and it is worth
reading twice: on two square operands both readings type-check and return the
right shape, so mistaking one for the other gives a wrong answer rather than an
error. `A * B` and `A @ B` are the same call.

With a scalar on either side, `*` and `/` scale every element as they always
have: `A * 2.0`, `2.0 * A`, `A / 2.0`. Only the matrix-by-matrix case changed.

### Element-wise product, quotient and power

`*` is taken, so these are methods rather than operators. Each is the method
form of the like-named routine, taking the same operands:

| Element-wise      | Method       | Routine        |
| ----------------- | ------------ | -------------- |
| Hadamard product  | `a.mul(b)`   | `mul(a, b)`    |
| Quotient          | `a.div(b)`   | `div(a, b)`    |
| Power             | `a.pow(b)`   | `pow(a, b)`    |

Each also takes a scalar: `a.pow(2.0)` squares every element, which is what
`a ** 2.0` used to do. `a.mul(2.0)` and `a.div(2.0)` are the same operations
as `a * 2.0` and `a / 2.0` and exist so the element-wise family reads as one
set.

There is no `A / B` between two matrices. Division by a matrix is a solve
under another name, and which side you are solving on is not something a
single character should decide - write [`solve`](#linear-algebra) and say so.

Every operator has a named routine behind it in `routines.math` and
`routines.logic` — `add`, `sub`, `mul`, `div`, `matmul`, `matrix_power`,
`floordiv`, `mod`, `pow`, `greater`, `equal`, and so on, plus a `scalar_*` form
of each for the matrix-and-scalar case. Use them when you want to be explicit,
or when the operator syntax will not fit.

Element-wise binary operations require identical shapes and raise `ValueError`
otherwise; `*`, `@` and `**` are the linear-algebra operations and check what
linear algebra requires — inner dimensions that agree, and a square operand for
`**`. There is no implicit broadcasting — stretch an operand yourself with
[`broadcast_to`](#shape-and-layout) when you want it.

### Matrix power

`A ** n` multiplies `A` by itself `n` times, by repeated squaring, so `A ** 13`
costs five products rather than twelve. The exponent is an `Int`: only a whole
number of multiplications is defined, and a fractional power would need an
eigendecomposition.

```mojo
a**3   # a @ a @ a
a**1   # a
a**0   # the identity matrix
a**-1  # inv(a)
a**-2  # inv(a) @ inv(a)
```

`A` must be square, and a negative exponent additionally needs it to be
invertible; both raise `ValueError` otherwise. On an arbitrary-precision
element type a negative exponent always raises, because `inv` is defined over
the SIMD element types only.

### Comparisons return masks

`a == b` is an element-wise `Matrix[Scalar[DType.bool]]` of the same shape, not
a single `Bool`. `Matrix` therefore does not conform to `EqualityComparable` on
purpose. To ask whether two matrices are wholly identical, use
`assert_matrices_equal` / `assert_matrices_close` from `utils/test_utils.mojo`,
or reduce the mask with `all`:

```mojo
from linamo.routines.logic import all
if all(a == b):
    ...
```

### Reflected operators

Scalars work on the left as well: `2.0 + A`, `2.0 * A`, `2.0 - A`, `2.0 / A`.
The subtraction and division forms keep the operand order you would expect --
`2.0 - A` subtracts each element from 2.0, not the reverse.

### In-place operators

`+=`, `-=`, `//=` and `%=` are defined on `Matrix` only, and accept a matrix, a
view, or a scalar. Unlike the out-of-place operators they allocate nothing:
they write back through the matrix's own strides, so a transposed or
column-major matrix keeps its layout.

`*=` and `/=` take a **scalar only**. A matrix on the right would have to mean
the matrix product, and that cannot be done in place: every element of the
target is read after the point where it would have been overwritten, so it
needs a full temporary. Write `a = a * b` and let the allocation show.

`MatrixView` has no in-place operators, for the same reason it has no `store`
method: the type is generic over its origin, and Mojo checks a method body
against the read-only instantiation too, so nothing that writes through
`self._data` can be defined on it. Mutate a view through the free functions in
`routines/mutation.mojo`.

Aliasing is a compile error rather than a silent wrong answer:

```mojo
a += a[:, :]   # does not compile
```

The borrow checker will not produce a mutable reference to `a` while a view
borrowing `a` is still alive. This is the same mechanism that makes views safe
in general - no runtime flag, no defensive copy.

---

## Mutating a matrix

An owned matrix has one write method, `set`, plus `store` for a SIMD run.
Single elements can also be written through indexing:

```mojo
var A = la.zeros[Float64](3, 3)

A[0, 0] = 1.0                              # one element, by subscript
A.set(0, 0, 1.0)                           # one element, by name
A.set(2.0)                                 # every element
A.set(src)                                 # a whole matrix copied in
A.set(Slice(0, 2), Slice(0, 2), 9.0)       # a region
A.set(Slice(0, 1), Slice(0, 3), src)       # a block copied into a region
A.store[2](0, 0, SIMD[DType.float64, 2](1.0))  # a SIMD run along a row
```

Everything beyond that goes through `linamo.routines.mutation`, the only
module that writes through a `MatrixView`:

```mojo
from linamo.routines.mutation import (
    view_mut, fill, assign, store, rows_mut, cols_mut,
)

var B = la.zeros[Float64](4, 4)

var v = B.view_mut(Slice(0, 2), Slice(0, 2))    # writable 2 x 2 view
fill(v, 5.0)                                    # the whole view
fill(v, Slice(0, 1), Slice(0, 2), 5.0)          # a region of it
assign(v, Slice(0, 1), Slice(0, 2), src.view())
store[2](v, 0, 0, SIMD[DType.float64, 2](7.0))

for row in rows_mut(B):     # each row is a writable 1 x ncols view
    row[0, 0] = 1.0
```

Three things follow from the signatures, and are worth stating plainly:

**A read-only view cannot be passed to any of them.** `fill`, `assign`, `store`
and the sub-view form of `view_mut` are pinned to `Origin[mut=True]`, so the
mistake is caught at the call site, in the compiler, not at run time.

**`view_mut` inherits the mutability of its source.** `m.view_mut(...)` is
writable when `m` is a `var` and read-only otherwise - including when `m` is a
temporary, so a view can never outlive what it points at. Mutability is only
ever inherited, never manufactured.

**A mutable view is an exclusive borrow.** It cannot appear twice in one
expression, and nothing else may read the matrix while it is alive. Demote it
with `as_imm()` when you need to read through it more than once.

The slice arguments are `Slice(start, stop)` values rather than `a:b` syntax,
because `a:b` is only legal inside `[]`. `Slice(a, b, step)` works too.

---

## Iteration

`len()` returns the number of rows, so it agrees with what iteration yields.
Use `size()` for the element count.

Iterating a matrix or a view walks its rows, yielding each as a `1 x ncols`
view onto the parent buffer. Nothing is copied.

```mojo
for row in A:                 # each row as a 1 x ncols view
    print(sum(row))

for col in A.cols():          # each column as an nrows x 1 view
    print(sum(col))

for row in A.rows[False]():   # bottom to top
    print(row)
```

`rows()` and `cols()` both take a `forward` parameter, so `rows[False]()` walks
last to first. Mojo's builtin `reversed()` only accepts specific
standard-library containers, so it will not dispatch to `__reversed__` on these
types; call `rows[False]()`, which is clearer at the call site anyway.

Rows yielded by `rows()`, `cols()` and `for ... in` are read-only. To write
through them, use `rows_mut` / `cols_mut` from
[`routines.mutation`](#mutating-a-matrix).

---

## Reductions, searches and sorts

Every routine here comes in two forms: without an axis it reduces the whole
matrix, and with one it reduces along that axis.

**`axis` names the dimension that disappears.** `axis=0` collapses the rows and
returns a `1 x ncols` result; `axis=1` collapses the columns and returns
`nrows x 1`. This is NumPy's convention. An axis other than 0 or 1 raises
`ValueError`.

| Routine             | Module               | Whole matrix        | With `axis`                    |
| ------------------- | -------------------- | ------------------- | ------------------------------ |
| `sum`, `prod`       | `statistics`, `math` | `Scalar[d]`         | `Matrix[Scalar[d]]`            |
| `min`, `max`        | `math`               | `Scalar[d]`         | `Matrix[Scalar[d]]`            |
| `cumsum`, `cumprod` | `statistics`, `math` | same shape, scanned | same shape, scanned            |
| `argmin`, `argmax`  | `searching`          | `Int`, row-major    | `Matrix[Int64]`                |
| `all`, `any`        | `logic`              | `Bool`              | `Matrix[Scalar[DType.bool]]`   |
| `sort`              | `sorting`            | —                   | axis required                  |
| `argsort`           | `sorting`            | —                   | `Matrix[Int64]`                |
| `sort_inplace`      | `sorting`            | —                   | axis required, writes `Matrix` |

```mojo
from linamo.routines.statistics import sum, cumsum
from linamo.routines.math import min, max, prod
from linamo.routines.searching import argmax
from linamo.routines.logic import all, any
from linamo.routines.sorting import sort, sort_inplace

var A = la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(sum(A))          # 21.0
print(sum(A, 0))       # 1 x 3:  [5.0, 7.0, 9.0]
print(sum(A, 1))       # 2 x 1:  [6.0], [15.0]
print(max(A))          # 6.0
print(argmax(A))       # 5  — row-major index of the largest element
print(all(A > 0.0))    # True

var S = sort(A, 1)     # each row sorted, A untouched
sort_inplace(A, 1)     # each row sorted, in place, layout preserved
```

`cumsum` and `cumprod` without an axis read the matrix as if flattened
row-major and return a result of the *same shape*, so it can be read back with
the original indices.

`argmin` and `argmax` break ties towards the first occurrence, as NumPy does.
`argsort` is stable: equal elements keep their relative order.

Reducing an empty matrix gives zero for `sum` and one for `prod`; `min` and
`max` raise `ValueError`, because there is no answer to give.

All of these accept a `Matrix` or a `MatrixView` — including a strided one —
and a routine that works on `A` works unchanged on `A[0:4:2, 1:5:2]`.

---

## Custom reductions

Two building blocks in `routines.functional` cover the reductions the library
does not ship.

`fold` threads an accumulator over every element in memory order:

```mojo
from linamo.routines.functional import fold

def _add(a: Float64, b: Float64) -> Float64:
    return a + b

var total = fold[func=_add](A.view(), 0.0)
```

`apply_along_axis` hands each lane to a kernel as a *view* onto the original
buffer — nothing is copied, and a lane of a strided matrix is itself just a
strided view — and collects one scalar per lane:

```mojo
from linamo.routines.functional import apply_along_axis

def _count_positive[
    d: DType, origin: Origin[mut=False]
](v: la.MatrixView[Scalar[d], origin]) -> Scalar[d]:
    var n = Scalar[d](0)
    for i in range(v.nrows()):
        for j in range(v.ncols()):
            if v[i, j] > 0:
                n += 1
    return n

var counts = apply_along_axis[
    axis=1, func=_count_positive[DType.float64, type_of(A.view()).origin]
](A.view())
```

`axis` is a compile-time parameter, so the traversal is specialised rather than
branched at run time, and the kernel is a compile-time parameter too, so it
inlines. The kernel's origin has to name the buffer it will be handed, and
`type_of(A.view()).origin` is how a caller spells it: the origin that a view of
`A` carries, without naming `A`'s buffer directly.

---

## Shape and layout

### The two layouts

A `Matrix` stores its elements row-major (C-contiguous) or column-major
(Fortran-contiguous), chosen with `order` at creation. The layout changes where
the elements sit in memory and nothing about what `A[i, j]` means.

Shape and layout are readable at any time:

| Query                          | Answer                                    |
| ------------------------------ | ----------------------------------------- |
| `nrows()`, `ncols()`           | the shape                                 |
| `size()`                       | `nrows() * ncols()`                       |
| `len(m)`                       | `nrows()`, to agree with iteration        |
| `row_stride()`, `col_stride()` | the distance in memory between neighbours |
| `offset()`                     | where the view starts in the buffer       |
| `is_c_contiguous()`            | dense and row-major                       |
| `is_f_contiguous()`            | dense and column-major                    |
| `is_row_contiguous()`          | neighbours along a row are adjacent       |
| `is_col_contiguous()`          | neighbours down a column are adjacent     |

These are methods, not fields. The underlying `_nrows`, `_ncols`, the two
strides and `_data` are private, because they are one invariant bundle rather
than five independent numbers: assigning to any one alone leaves a matrix that
indexes outside its own buffer. Mojo 1.0 has no access control and no
properties, so the leading underscore is the marker and the parentheses are the
cost of having one. Every accessor is `@always_inline`; the layer does not
survive into the generated code.

`StaticMatrix` spells them the same way, `m.nrows()` and `m.row_stride()`,
even though there they read a struct parameter and a `comptime` alias rather
than a field. Naming them alike is what lets one piece of read-only code take
any of the three; the accessors are `@always_inline` over compile-time
constants, so the parentheses cost nothing.

The last two are the weaker tests, and they are the ones the kernels use: a
lane taken out of a larger matrix has unit stride along its own extent while
saying nothing about the other axis.

Views are where layout stops being a simple flag. A slice with a step, or a
sub-block, is neither C- nor F-contiguous, and the library never requires it to
be. Every routine accepts a strided view; the kernels branch on contiguity once
and take a slower path when they must, so only the speed changes.

### Reshaping routines

| Routine                               | Copies? | Result                                                              |
| ------------------------------------- | ------- | ------------------------------------------------------------------- |
| `reshape(a, nrows, ncols, order="C")` | yes     | a new C-contiguous matrix, elements read and written in `order`     |
| `reshape_view(a, nrows, ncols)`       | no      | a view of the same buffer under a new shape; requires a dense input |
| `flatten(a, order="C")`               | yes     | a new `1 x size` matrix                                             |
| `resize(a, nrows, ncols)`             | yes     | truncated or zero-padded to the new shape                           |
| `contiguous(a, order="C")`            | yes     | a dense copy in the requested layout                                |
| `reorder_layout(a)`                   | yes     | a copy in the opposite layout                                       |
| `broadcast_to(a, nrows, ncols)`       | no      | size-1 dimensions stretched by zero strides                         |
| `astype[target](a)`                   | yes     | a C-contiguous copy cast to `target`                                |
| `transpose(a)`                        | yes     | a new matrix with the axes exchanged                                |

```mojo
var A = la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

var R = la.reshape(A, 3, 2)          # copies
var V = la.reshape_view(A.view(), 3, 2)  # no copy, shares A's buffer
var F = la.flatten(A)                # 1 x 6
var C = la.contiguous(A[0:2:1, 0:3:2], "C")  # densify a strided view
var I = la.astype[Int32](A)    # truncates towards zero
var B = la.broadcast_to(A[0:1, :], 4, 3)  # one row read as four
```

`reshape` copies and `reshape_view` does not, and the difference is not a
performance footnote: `reshape_view` returns a view carrying the *same origin*,
so the matrix behind it is kept alive as long as the view is, and it requires a
dense input because only then does "the elements in memory order" mean the same
thing before and after. A strided input raises `ValueError` — pass it through
`contiguous` first.

`broadcast_to` stretches an extent-1 dimension by giving it a stride of zero,
so every index along it lands on the same element. It costs nothing and shares
its buffer with the source. The result is read-only, as in NumPy, and for a
stronger reason here: several logical positions map onto one element, so a
write would be visible in places the caller did not name.

`astype` uses Mojo's `SIMD.cast`, so the usual rules apply — float to integer
truncates towards zero, and a narrowing conversion wraps.

---

## Linear algebra

Everything in `routines.linalg` accepts a `Matrix` or a `MatrixView` on either
side and returns owning matrices.

| Routine              | Returns              | Notes                                    |
| -------------------- | -------------------- | ---------------------------------------- |
| `transpose(a)`       | `Matrix`             | a new matrix, axes exchanged             |
| `trace(a)`           | `Scalar[d]`          | square input required                    |
| `det(a)`             | `Scalar[d]`          | via LU                                   |
| `inv(a)`             | `Matrix`             | via LU; solves `A @ X = I`               |
| `matrix_power(a, n)` | `Matrix`             | repeated squaring; what `a ** n` calls   |
| `lu(a)`              | `(L, U, piv)`        | partial pivoting, `PA = LU`              |
| `cholesky(a)`        | `Matrix` (lower `L`) | symmetric positive-definite input        |
| `qr(a)`              | `(Q, R)`             | Householder reflections, `m >= n`        |
| `solve(A, b)`        | `Matrix`             | via LU; `b` may have several columns     |
| `lstsq(A, b)`        | `Matrix`             | via QR; overdetermined systems, `m >= n` |

```mojo
var A = la.matrix[Float64](
    [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
)
var b = la.matrix[Float64]([[1.0], [2.0], [3.0]])

var x  = la.solve(A, b)
var Ai = la.inv(A)
var d  = la.det(A)
var L  = la.cholesky(A)

var lup = la.lu(A)
var Lu = lup[0].copy()
var Uu = lup[1].copy()
var piv = lup[2].copy()

var qr_result = la.qr(A)
var Q = qr_result[0].copy()
var R = qr_result[1].copy()
```

The tuple results are unpacked with an index and `.copy()`, because `Matrix` is
not implicitly copyable and taking it out of the tuple is a copy you have to
ask for.

A non-square input raises `ValueError` from every routine that needs a square
one, and `cholesky` raises if the input is not positive definite.

A singular matrix is answered differently by the two element paths, because
they divide differently. On a scalar element the division is IEEE arithmetic:
`det` returns zero and `inv`, `solve` and a negative `matrix_power` return
infinities, without raising. On an arbitrary-precision element there is no
infinity to return, so `det` still reports zero but `inv`, `solve` and a
negative `matrix_power` raise `ValueError` naming the matrix as singular.

`trace`, `det`, `lu`, `solve`, `inv` and `matrix_power` all carry a second
overload for the arbitrary-precision element types; `cholesky`, `qr` and
`lstsq` are scalar-only, since they need a square root. See
[Arbitrary-precision elements](#arbitrary-precision-elements) for what division
means there.

---

## SIMD access

`load[width](row, col)` reads a run of elements along a row, and returns them
as a SIMD vector. When the row is contiguous (`col_stride == 1`) this is a
single vector load; on a strided view it gathers element by element. The call
is always correct, and only the speed changes. Widths must be powers of two.

Writing works the same way but lives in different places, for a reason worth
knowing about. `Matrix.store[width]` is an ordinary method, because a `Matrix`
owns a concretely mutable buffer. A `MatrixView`, though, is generic over its
origin, and Mojo type-checks a method body against every instantiation --
including the read-only one - so no method that writes through `self._data`
can even be defined. Bulk writes on views are therefore free functions in
`routines/mutation.mojo`, pinned to a mutable origin:

- `store[width](view, row, col, value)` - write a SIMD run.
- `fill(view, value)` - write one scalar across the whole view.
- `fill(view, rows, cols, value)` - write one scalar across a region.
- `assign(view, src)` - copy a block over the whole view.
- `assign(view, rows, cols, src)` - copy a block into a region.

The whole-view and region forms mirror `Matrix.set`, and `fill(view, value)` is
non-raising for the same reason `m.set(value)` is: it visits every index of the
view and none of them can be out of range.

Passing a read-only view to any of them fails to compile at the call site,
which is the guarantee we wanted. Single-element writes need none of this:
`v[i, j] = x` writes through the reference returned by `__getitem__`, where the
caller's origin is already concrete.

`Matrix` needs none of that, so it carries a single `set` method instead,
overloaded on its arguments. `set` delegates to the functions above rather than
looping over `self._data` itself, so each write has exactly one implementation
in the library.

Note that region assignment is spelled as a named method rather than
`m[a:b, c:d] = src`. Mojo routes slice-assignment through `__getitem__`, which
would force the right-hand side to be a view carrying the *target's* own
origin - making assignment from any other matrix inexpressible.

---

## NumPy interoperability

```mojo
from linamo import from_numpy, to_numpy
from std.python import Python

def main() raises:
    var np = Python.import_module("numpy")
    var arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    var A = from_numpy[Float64](arr)   # numpy -> Linamo
    var back = to_numpy(A)                   # Linamo -> numpy
```

Both directions **copy**. The input array must be 2-D and non-empty, and the
results are always C-contiguous regardless of the source's layout. The dtype is
not inferred from the array — name it as a parameter, and the conversion raises
if the array cannot supply it.

There is no `to_ndarray` for NuMojo. NuMojo's own `Matrix` type is gone, and
reproducing its N-dimensional array here would mean depending on it. You do not
need one to use Linamo on a NuMojo array, though — see
[NuMojo interoperability](#numojo-interoperability), which borrows the array's
buffer instead of copying it.

---

## NuMojo interoperability

Linamo does not depend on NuMojo and ships no bridge, but it does not need one.
`MatrixView` is parametric over its origin and holds nothing but a `Span`, a
shape, two strides and an offset, so it can view *any* allocation — including
one NuMojo owns. The two functions below are the entire bridge. Paste them into
your own project, where both packages are on the import path; they copy nothing
and are `O(1)`.

```mojo
import numojo as nm
from linamo import MatrixView

def as_matrix_view[
    d: DType
](imm a: nm.NDArray[d]) raises -> MatrixView[Scalar[d], origin_of(a)]:
    """Read-only 2-D view of a NuMojo array. No data is copied."""
    if a.ndim != 2:
        raise Error("as_matrix_view: expected a 2-D NDArray")
    return MatrixView[Scalar[d], origin_of(a)](
        buffer=Span[Scalar[d], origin_of(a)](
            unsafe_ptr=a._buf.get_ptr()
            .as_imm()
            .unsafe_origin_cast[origin_of(a)](),
            length=len(a._buf),
        ),
        nrows=Int(a.shape[0]),
        ncols=Int(a.shape[1]),
        row_stride=Int(a.strides[0]),
        col_stride=Int(a.strides[1]),
        offset=a.offset,
    )

def as_matrix_view_mut[
    d: DType
](mut a: nm.NDArray[d]) raises -> MatrixView[Scalar[d], origin_of(a)]:
    """Writable 2-D view of a NuMojo array. No data is copied."""
    if a.ndim != 2:
        raise Error("as_matrix_view_mut: expected a 2-D NDArray")
    return MatrixView[Scalar[d], origin_of(a)](
        buffer=Span[Scalar[d], origin_of(a)](
            unsafe_ptr=a._buf.get_ptr().unsafe_origin_cast[origin_of(a)](),
            length=len(a._buf),
        ),
        nrows=Int(a.shape[0]),
        ncols=Int(a.shape[1]),
        row_stride=Int(a.strides[0]),
        col_stride=Int(a.strides[1]),
        offset=a.offset,
    )
```

The two libraries line up field for field, which is why this is so short.
NuMojo's strides and `offset` are counted in *elements*, exactly as Linamo's
are, and its buffer pointer is already a `Pointer[Scalar[d], ...]` — the one
thing `Span`'s unsafe constructor asks for.

Use them like any other view:

```mojo
from linamo import det, inv
from linamo.routines.mutation import assign

def main() raises:
    var a = nm.array[nm.f64]("[[4.0, 7.0], [2.0, 6.0]]")
    var v = as_matrix_view(a)

    print(det(v))          # 10.0
    print(inv(v))
    print(v @ v)

    # Send a result back into NuMojo memory without a copy out:
    var out = nm.zeros[nm.f64](nm.Shape(2, 2))
    var ov = as_matrix_view_mut(out)
    assign(ov, v @ v)
```

Reach for `as_matrix_view` by default and `as_matrix_view_mut` only to write.
That is not merely stylistic: around thirty routines — `sum`, `cumsum`, the
`logic` predicates, the searches and sorts — bind their operand as
`Origin[mut=False]` and will not accept a writable view at all. A mutable view
reaches them through `.as_imm()`, but taking the read-only overload up front is
simpler.

### Contiguity is not required

A NuMojo array does **not** have to be contiguous. Every Linamo kernel is
stride-aware, and the `is_c_contiguous` / `is_row_contiguous` tests scattered
through `math`, `logic` and `manipulation` select a faster path rather than
gate a correct one — `matmul` alone has four, the last of which is a general
stride-aware loop that any layout lands on. A view with `row_stride = 8` and
`col_stride = 2` gives the same answers as the densified copy of itself.

Layout is therefore a performance question, and the answer usually favours
densifying. Multiplying two 256×256 views carved column-wise out of a wider
buffer:

| Path                                     | Time     |
| ---------------------------------------- | -------- |
| `matmul` on the strided views directly   | ~2800 µs |
| `contiguous()` on both operands          | ~260 µs  |
| `matmul` on the densified copies         | ~1100 µs |

The copy costs about a quarter of one dense multiply and removes about 1700 µs
from it, so it repays itself before the first operation finishes. Call
`contiguous()` once when a non-contiguous array is about to see real work, and
skip it for a one-off `det` or `sum`.

### The one rule: do not reallocate

While a view is alive, the array it borrows from must keep the same buffer.
Changing the *values* is fine — that is what `as_matrix_view_mut` is for, and
NuMojo and Linamo will simply see each other's writes. What must not happen is
anything that frees or moves the allocation: `resize()` above all, but equally
letting the array go out of scope or moving out of it with `^`.

Mojo catches some of this. Because the view carries `origin_of(a)` rather than
an untracked origin, moving the array out from under a live view is a compile
error:

```console
error: use of uninitialized value 'a'
```

But it does not catch all of it, and the gap is quiet:

```mojo
var out = nm.zeros[nm.f64](nm.Shape(2, 2))
var ov = as_matrix_view_mut(out)
out.resize(nm.Shape(4, 4))    # reallocates; `ov` now points at freed memory
assign(ov, matmul(v, v))      # writes into the freed block — silently lost
```

This compiles, runs, prints no warning, and `out` ends up all zeros. An origin
tracks the array *handle*, not the heap block behind it, so a reallocation slips
underneath it. The same hole exists for `Matrix` and is described in
[Appendix A](#appendix-a-how-it-works-inside); Linamo's own code avoids it by
never growing a buffer that a view might be borrowing, and code using this
bridge has to make the same promise by hand.

The narrow rule, then: **treat the NuMojo array as frozen in shape for as long
as the view lives.** Read it, write through it, but do not resize it, and let
the view die before the array does.

---

## Errors

Linamo raises Mojo `Error` values built by constructors in `types/errors.mojo`,
which format themselves as a Python-style traceback:

```console
Traceback (most recent call last):
  File "./src/linamo/routines/math.mojo", line 197, in _elementwise_view()
ValueError: Input matrices must have the same shape.
```

The file and line are captured at the raise site, and the absolute path is
shortened to a `./`-relative one so a traceback does not leak the build
machine's directory layout.

| Constructor         | Raised when                                       |
| ------------------- | ------------------------------------------------- |
| `ValueError`        | shapes disagree, an axis is not 0 or 1,           |
|                     | an `order` is not `"C"`/`"F"`,                    |
|                     | a matrix is singular or not positive definite     |
| `IndexError`        | an index or a SIMD run leaves the matrix          |
| `ZeroDivisionError` | division by zero where it is detectable           |
| `ConversionError`   | a value cannot be converted to the requested type |
| `OverflowError`     | an operation overflows its element type           |
| `KeyError`          | a lookup fails                                    |

These are constructor *functions* returning a plain `Error`, not distinct
types, so catching is `except e:` and inspection is on the message. Mojo has no
typed exceptions.

Almost every public routine is `raises`, because shape checking is a runtime
matter. Errors that can be caught at compile time — writing through a read-only
view, mismatched dtypes, a bad `axis` passed to `apply_along_axis` — are
compile errors instead, and are not in this table.

---

## Arbitrary-precision elements

`Matrix` takes an element type, so `Matrix[BInt]`, `Matrix[Decimal]` and
`Matrix[Dec128]` are ordinary matrices holding
[Decimo](https://github.com/forfudan/decimo) numbers. They are not a separate
API: the operators, the creation routines and the reductions are the same
names, and Linamo re-exports the element types so that using them costs no
second import.

```mojo
import linamo as la
from linamo import BInt

var a = la.matrix[BInt]([[1, 2], [3, 4]])

a + a               # element-wise, as for any other element type
a @ a               # matrix multiplication, and `a * a` is the same call
a**2                # repeated multiplication; `a**-1` inverts, see below
a.mul(a)            # element-wise, the Hadamard product
a * BInt(3)         # a value on either side of the operator
a.transpose()       # only moves elements
la.sort(a, 1)       # only compares them
la.trace(a)         # the diagonal sum
la.eye[BInt](3)     # asks `Numeric` for a zero and a one

la.from_string[BInt]("[[1, 2], [3, 4]]")   # asks `Parsable` to read the text
```

Elimination is there too — `lu`, `det`, `solve`, `inv`, and `a**-1` through
them — but it wants a *decimal* element:

```mojo
import linamo as la
from linamo import Dec128

var m = la.from_string[Dec128]("[[4, 7], [2, 6]]")

la.det(m)                # 10
la.inv(m)                # 0.6 -0.7 / -0.2 0.4, exactly
m ** -1                  # the same matrix, through `matrix_power`
la.solve(m, la.from_string[Dec128]("[[1], [1]]"))
```

These four divide, so they mean whatever `/` means on the element type.
`Decimal128` and `BigDecimal` give a quotient rounded to the type's precision,
which is an ordinary approximate answer carrying more digits than `Float64`
would — a `BigDecimal` inverse is *not* exact, it is exact to 28 significant
digits. `BigInt` truncates toward zero, and an integer matrix has no integer
inverse in general, so elimination over `BigInt` compiles, runs, raises
nothing, and returns whole numbers that are not the answer. Nothing in the
signature stops you; a decimal element type is what these routines are for.

The element types Linamo re-exports are exactly those that conform to
`decimo.Numeric`, under both of Decimo's spellings. All three also conform to
`decimo.Parsable`, which is what `from_string` above asks for — the two traits
are separate because the capabilities are, and a routine asks for whichever it
uses:

| Long name    | Short name | What it is                                    |
| ------------ | ---------- | --------------------------------------------- |
| `BigInt`     | `BInt`     | arbitrary-precision integer (also `Integer`)  |
| `BigDecimal` | `BDec`     | arbitrary-precision decimal (also `Decimal`)  |
| `Decimal128` | `Dec128`   | 128-bit exact decimal                         |

Decimo's `BigUInt`, `Rational` and `BigFloat` are deliberately not re-exported.
They do not conform to `Numeric`, so a matrix of them would have no arithmetic,
and naming them here would promise one. Import them from `decimo` directly if
you want to store them.

### How it works, and what it costs

Every routine name above has two overloads. The scalar one is selected by
`Self.T == Scalar[d]` and runs the SIMD kernels; the other is selected by
`conforms_to(Self.T, Numeric)` and runs a plain loop. No element type
satisfies both clauses, so `a + b` is one spelling that resolves to whichever
applies, and neither overload can shadow the other.

The split exists because a `Scalar[d]`'s `+` lowers to a vector instruction
that an arbitrary-precision element has no equivalent of. There is therefore no
SIMD and no `parallelize` on the `Numeric` path: a `BInt` addition allocates,
so those loops are memory-bound and the plain triple loop in `matmul` is what
the operation costs.

`//` and `%` have no `Numeric` counterpart. The trait closes over `+ - * /` and
nothing else, and `/` on an integral element truncates toward zero the way
`Int` does. `**` does carry over, since a matrix power is repeated
multiplication.

The elimination routines ask for `Numeric & Comparable` rather than `Numeric`
alone: partial pivoting has to rank candidate pivots by magnitude. All three
re-exported element types are `Comparable`, so nothing is excluded in practice.

### Decimo is a hard dependency

The trait lives in Decimo rather than here because Mojo's conformance is
nominal and has to be declared where the struct is defined: only Decimo can say
that `BigInt` is `Numeric`. Since Linamo's matrix types name that trait, and
Mojo has no conditional imports, Decimo has to be on the include path to
compile any part of Linamo:

```bash
pixi run decimo    # build the decimo package into temp/
```

`pixi run test`, `pixi run examples` and `pixi run pack` depend on that task,
so running them is enough.

## StaticMatrix

`StaticMatrix[T, num_rows, num_cols]` carries its shape in its type and stores
its elements in a `SIMD` register buffer rather than on the heap. Nothing is
allocated, and the shape is known to the optimiser.

```mojo
var S = la.smatrix[2, 3, Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(S.size())            # 6
print(S[1, 2])             # 6.0
print(S.is_c_contiguous()) # False — see below
```

The buffer is padded to the next power of two in each dimension, so a `2 x 3`
matrix occupies a `2 x 4` SIMD vector and a `3 x 3` occupies `4 x 4`. That is
what makes the register storage possible, and it is why the type suits small
fixed-size matrices — the 2×2, 3×3 and 4×4 of geometry — rather than large
ones. It is also why `is_c_contiguous()` is False unless `num_cols` is already
a power of two: the row stride is the padded width, not `num_cols`.

`StaticMatrix` is the least developed of the three types. It has the shape and
layout queries, element reads, printing, and `+` and `@`. It does not have the
rest of the operator set, element writes, slicing, views, or any of the
routines. Use `Matrix` for anything beyond small fixed-size storage and those
two operations.

### Crossing over to `Matrix`

A `StaticMatrix` shares no operator with `Matrix` or `MatrixView`, so `M + S`
does not compile. `S.to_matrix()` copies it into a freshly allocated `Matrix`,
and from there the whole library applies:

```mojo
var S = la.smatrix[2, 2, Float64]([[1.0, 2.0], [3.0, 4.0]])
var M = la.matrix[Float64]([[10.0, 10.0], [10.0, 10.0]])

print(M + S.to_matrix())      # 11 12 / 13 14
print(la.sum(S.to_matrix()))  # 10.0
```

The name matches `MatrixView.to_matrix()`, which does the same job
for the other non-owning type: walk a source whose layout is not dense and
produce owned C-contiguous storage. The power-of-two padding does not survive
the copy.

The conversion is deliberately explicit rather than an `@implicit` constructor.
Making it implicit would let a wrong-shaped `StaticMatrix` reach the dynamic
kernels, and `S2x2 + S3x3` --- which the compiler rejects today, because the
shapes are in the type --- would become a runtime `ValueError`. Naming the hop
is what keeps that check.

---

## Appendix A: how it works inside

Nothing here is needed to use the library. It is here because the shape of the
public API follows from it, and because a reader who wonders why every routine
takes a `MatrixView` deserves an answer.

### `Matrix` and `MatrixView` inter-operate

Any binary routine or operator accepts either type on either side, and the
result is always a **new** `Matrix` that owns its data.

That gives four operand permutations per operation --- `(M, M)`, `(M, V)`,
`(V, M)`, `(V, V)` --- and each of those can be contiguous or strided. Writing
eight bodies per operation would be unmaintainable, so the library collapses
both dimensions before any real work happens. Three layers, and only the
innermost one contains an algorithm:

| Layer         | Lives in                        | Job                                     |
| ------------- | ------------------------------- | --------------------------------------- |
| Operators     | `types/matrix{,_view}.mojo`     | `a + b` to `add(a, b)`                  |
| Named routine | `routines/*.mojo`, public       | any operand pair to a pair of views     |
| Kernel        | `routines/*.mojo`, `_`-prefixed | the actual loop, layout dispatch inside |

**The permutations collapse at the call site, not in the library.** Every
public routine takes `MatrixView` operands and nothing else --- one signature,
not four:

```mojo
def add[d: DType, origin_a: Origin, origin_b: Origin](
    a: MatrixView[Scalar[d], origin_a], b: MatrixView[Scalar[d], origin_b]
) raises -> Matrix[Scalar[d]]:
    return _elementwise_view[func = Scalar[d].__add__](a, b)
```

`add(a, b)` still compiles when either operand is a `Matrix`, because
`MatrixView` carries an `@implicit` constructor from `Matrix` and the compiler
inserts the conversion. It is the same O(1) metadata copy `view()` performs, so
nothing is allocated and nothing is copied.

Two details make that constructor safe, and both are load-bearing:

```mojo
@implicit
def __init__[d: DType](
    out self: MatrixView[d, ImmOrigin(origin_of(m._data))], ref m: Matrix[d]
):
```

The argument is `ref m`, and only `ref` binds the origin to the *caller's*
storage. Under `imm`, `read` or the default convention the argument gets its
own origin --- `origin_of(m._data)` then names the callee's parameter slot ---
so the target type is one no caller can name, and every call site fails to
convert. And the result is wrapped in `ImmOrigin(...)`, so a `var` matrix
converts to a **read-only** view. Without that, `add(a, a)` would be two
mutable borrows of one matrix and would not compile --- the same wall that
forced slicing to become read-only. It also keeps `routines.mutation` the only
door to a mutable view: those signatures are pinned to `Origin[mut=True]`,
which this conversion can never satisfy, so `fill(m, ...)` remains a compile
error.

A view is the *general* case and a matrix the special one, so everything is
funnelled towards views rather than away from them. Below this line no code
knows or cares which of the four permutations the user wrote.

**The operator layer collapses the same way.** A dunder is a method, so its
left operand is fixed by the receiver's type: `Matrix.__add__` and
`MatrixView.__add__` both have to exist. The right operand is not fixed, and
both declare it as a `MatrixView`, so the same implicit conversion covers the
rest --- `A + B`, `A + V`, `V + B` and `V + V` reach two overloads between
them rather than four. The only binary operator needing a second signature is
the scalar form (`A + 2.0`), because a scalar is not a matrix and does not
convert to one.

**The layouts collapse inside the kernel.** A kernel branches on contiguity
once, up front, and then runs a loop that has no further tests in it:

```mojo
if a.is_c_contiguous() and b.is_c_contiguous():
    # one flat SIMD sweep over nrows * ncols
else:
    # index through row_stride / col_stride
```

Both branches are always correct; only the speed differs. This is why a
strided view is never a special case a caller has to think about --- a routine
that works on `a` works unchanged on `a[0:8:2, 1:9:2]`.

**The kernels are parameterised by the element operation.** `add`, `sub`,
`mul`, `div`, `floordiv`, `mod` and `pow` share a single body,
`_elementwise_view`, which takes the scalar function as a compile-time
parameter and is specialised per operation at compile time:

```mojo
_elementwise_view[func = Scalar[d].__add__](a, b)
_elementwise_view[func = Scalar[d].__mul__](a, b)
```

The kernels currently in use are:

| Kernel                        | Shape                                  |
| ----------------------------- | -------------------------------------- |
| `_elementwise_view`           | view, view to new matrix               |
| `_scalar_elementwise_view`    | view, scalar to new matrix             |
| `_elementwise_inplace`        | matrix, view, writes into the matrix   |
| `_scalar_elementwise_inplace` | matrix, scalar, writes into the matrix |
| `_compare_view`               | view, view to bool mask                |
| `_scalar_compare_view`        | view, scalar to bool mask              |
| `_matmul_view_simd`           | view, view to new matrix               |

`_matmul_view_simd` is the one kernel that dispatches on more than "contiguous
or not": it picks between four loop orders depending on which operand is row-
or column-contiguous, because for matrix multiplication the layout changes the
algorithm and not just the addressing.

The in-place kernels are the only ones without a `MatrixView` counterpart, and
that is a language constraint rather than a choice --- see
[In-place operators](#in-place-operators).

### Why matmul has several implementations

The memory layout of the operands changes which loop order touches memory in
order, so matrix multiplication is written more than once:

1. `c@c`: both matrices row-major.
2. `f@f`: both column-major.
3. `c@f`, `f@c`: mixed.
4. `c@v`, `f@v`, `v@c`, `v@f`: against a non-contiguous view, which falls back
   to an implementation that assumes no layout at all.

The dispatch happens once, before the loop, so the inner loop stays free of
tests.

---

## Appendix B: what is not here yet

This manual documents what exists. The
[roadmap](ROADMAP.md) tracks what does not; the larger gaps as of this writing
are:

- **Creation**: `randn`, for normally distributed random matrices.
- **Element-wise mathematics**: the trigonometric and hyperbolic functions,
  `round`, `isclose`/`allclose`, the infinity predicates and the logical
  operators.
- **Linear algebra**: `issymmetric`, an LU-based `solve_lu`, and eigenvalues.
- **Packaging**: an install path that is not `-I src`.

`StaticMatrix` is likewise a partial type; see
[StaticMatrix](#staticmatrix).
