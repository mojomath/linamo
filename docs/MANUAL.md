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
    - [The element type is a `DType`](#the-element-type-is-a-dtype)
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
  - [Indexing and slicing](#indexing-and-slicing)
  - [Copying and assignment](#copying-and-assignment)
  - [Operators](#operators)
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
  - [Errors](#errors)
  - [StaticMatrix](#staticmatrix)
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

Two import styles work, and the manual uses the first:

```mojo
import linamo as la          # la.matrix, la.zeros, la.transpose, ...
```

```mojo
from linamo.prelude import *  # the same, plus dtype aliases: float64, int32, ...
```

The prelude adds short names for the element types, so `la.matrix[float64]`
reads the same as `la.matrix[DType.float64]`. Everything in the top-level
namespace is re-exported from `src/linamo/__init__.mojo`; anything not there is
reached by its module path, for example
`from linamo.routines.mutation import fill`.

A first program:

```mojo
import linamo as la

def main() raises:
    var A = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var B = la.eye[DType.float64](2)
    print(A @ B)
    print(la.det(A))
```

### The element type is a `DType`

Every matrix carries its element type as a compile-time `DType` parameter, the
way `SIMD` does: `Matrix[DType.float64]`, `Matrix[DType.int32]`,
`Matrix[DType.bool]`. Elements are `Scalar[dtype]`, which is `SIMD[dtype, 1]`.
The parameter defaults to `DType.float64` almost everywhere, so `la.zeros(2, 3)`
is a float64 matrix.

The default is *not* available on the routines that take a list, though —
`la.matrix([[1.0, 2.0]])` does not compile, and `la.matrix[DType.float64](...)`
is required. The reason is that a list literal has no type of its own until it
is given one, and here it would have to be given `List[List[Scalar[dtype]]]`
with `dtype` still unknown; the compiler cannot resolve either side first. Name
the dtype, or give the list a type of its own first:

```mojo
var rows: List[List[Float64]] = [[1.0, 2.0], [3.0, 4.0]]
var A = la.matrix(rows)          # dtype inferred from the argument
```

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
var a = la.matrix[DType.float64]([[10.0, 20.0], [1.0, 2.0]])

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

| Call                                                       | Gives you                                     |
| ---------------------------------------------------------- | --------------------------------------------- |
| `matrix[dtype](list_of_rows, order="C")`                   | a matrix from nested lists                    |
| `matrix[dtype](flat_list=..., nrows=, ncols=, order=)`     | a matrix from one flat list                   |
| `zeros[dtype](nrows, ncols)`                               | all zeros                                     |
| `ones[dtype](nrows, ncols)`                                | all ones                                      |
| `full[dtype](nrows, ncols, fill_value)`                    | one repeated value                            |
| `eye[dtype](n)` / `identity[dtype](n)`                     | the `n x n` identity                          |
| `diag[dtype](values)`                                      | a square matrix with `values` on the diagonal |
| `diag[dtype](m)`                                           | the diagonal of `m`, as a `List`              |
| `smatrix[nrows, ncols, dtype](list_of_rows)`               | a `StaticMatrix`                              |
| `empty[dtype](nrows, ncols)`                               | uninitialised storage of that shape           |
| `zeros_like(m)` / `ones_like(m)`                           | zeros or ones shaped like `m`                 |
| `full_like(m, fill_value)` / `empty_like(m)`               | one value, or uninitialised, shaped like `m`  |
| `arange[dtype](stop)` / `arange[dtype](start, stop, step)` | a `1 x n` row of evenly spaced values         |
| `linspace[dtype](start, stop, num, endpoint)`              | a `1 x num` row from `start` to `stop`        |
| `fromlist[dtype](flat_list, nrows, ncols, order)`          | a matrix from one flat list, positionally     |
| `fromstring[dtype](text)`                                  | a matrix parsed from a literal                |
| `rand[dtype](nrows, ncols, low, high)`                     | uniform random values                         |
| `seed(value)`                                              | pins `rand` for reproducibility               |

```mojo
import linamo as la

def main() raises:
    var A = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = la.matrix[DType.float64](
        flat_list=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], nrows=2, ncols=3
    )
    var F = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]], order="F")

    var Z = la.zeros[DType.float64](2, 3)
    var O = la.ones[DType.int32](2, 3)
    var C = la.full[DType.float64](2, 2, 7.5)
    var I = la.eye[DType.float64](3)
    var D = la.diag[DType.float64]([1.0, 2.0, 3.0])
```

`order` chooses the memory layout: `"C"` for row-major, `"F"` for column-major.
It changes where the elements sit in memory and nothing else — `A[i, j]` means
the same thing either way. See [Shape and layout](#shape-and-layout).

The list-taking forms raise `ValueError` on an empty list, on rows of different
lengths, on a `flat_list` whose length is not `nrows * ncols`, or on an `order`
other than `"C"` or `"F"`. The shape-only routines — `zeros`, `ones`, `full`,
`eye`, `identity` — cannot fail and are not `raises` at all.

The `[dtype]` is required whenever a list is passed and optional otherwise; see
[The element type is a `DType`](#the-element-type-is-a-dtype) for why.

### Ranges, shapes copied from another matrix, and random values

`arange` and `linspace` return a **`1 x n` row**, because Linamo has no 1-D type
and a row is what NumPy's 1-D result prints as. `reshape(x, n, 1)` gives the
column.

```mojo
var x = la.arange[DType.float64](5.0)              # 1x5: 0 1 2 3 4
var y = la.arange[DType.float64](1.0, 2.0, 0.25)   # 1x4: 1 1.25 1.5 1.75
var d = la.arange[DType.int64](10, 0, -3)          # 1x4: 10 7 4 1
var t = la.linspace[DType.float64](0.0, 1.0, 5)    # 1x5: 0 0.25 0.5 0.75 1
var h = la.linspace[DType.float64](0.0, 1.0, 5, endpoint=False)
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
var A = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
var Z = la.zeros_like(A)          # 2x3 of zeros
var S = la.zeros_like(A[0:2, 1:3])  # 2x2 — a view works too
var E = la.empty[DType.float64](2, 2)  # contents unspecified; write before reading
```

`rand` draws uniformly from the closed interval `[low, high]`, defaulting to
`[0, 1]`. It uses the standard library's global generator, so `seed(n)` makes a
run reproducible:

```mojo
la.seed(42)
var R = la.rand[DType.float64](2, 3)         # values in [0, 1]
var Q = la.rand[DType.float64](3, 3, -2.0, 2.0)
var K = la.rand[DType.int64](2, 2, 1, 6)     # integers, both bounds included
```

### Parsing a matrix from text

`fromstring` reads a literal in which elements are separated by whitespace or
commas and rows by nested brackets:

```mojo
var A = la.fromstring[DType.float64]("[[1, 2, 3], [4, 5.5, 6]]")  # 2x3
var B = la.fromstring[DType.float64]("[[1 2 3]\n [4 5 6]]")       # 2x3
var C = la.fromstring[DType.float64]("1 2 3")                     # 1x3, one row
var D = la.fromstring[DType.int32]("[1, 2, 3, 4]", 2, 2)          # shape given
```

A literal with no nesting is a single row. The second overload takes an
explicit `nrows`, `ncols` and `order`, and ignores the bracket structure
entirely — it reads every element it finds, in `order`, into the shape you
asked for. Unbalanced brackets, nesting more than two deep, rows of unequal
length, and a cell that is not a number all raise `ValueError`; the last of
these names the offending token.

`fromlist` is the same thing for a list you already have — the positional
spelling of the keyword-only `matrix(flat_list=..., nrows=..., ncols=...)`.

---

## Indexing and slicing

```mojo
var A = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

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

| Operator                    | Meaning                                   |
| --------------------------- | ----------------------------------------- |
| `+` `-` `*` `/`             | Element-wise arithmetic                   |
| `//` `%`                    | Element-wise floor division and modulo    |
| `**`                        | Element-wise power (**not** matrix power) |
| `@`                         | Matrix multiplication                     |
| `<` `<=` `>` `>=` `==` `!=` | Element-wise mask, `Matrix[DType.bool]`   |

`**` follows NumPy: `A ** 2` squares each entry. Matrix exponentiation is a
different operation and gets a named routine, not an operator.

Every operator has a named routine behind it in `routines.math` and
`routines.logic` — `add`, `sub`, `mul`, `div`, `matmul`, `floordiv`, `mod`,
`pow`, `greater`, `equal`, and so on, plus a `scalar_*` form of each for the
matrix-and-scalar case. Use them when you want to be explicit, or when the
operator syntax will not fit.

Element-wise binary operations require identical shapes and raise `ValueError`
otherwise; `@` requires the inner dimensions to agree. There is no implicit
broadcasting — stretch an operand yourself with
[`broadcast_to`](#shape-and-layout) when you want it.

### Comparisons return masks

`a == b` is an element-wise `Matrix[DType.bool]` of the same shape, not a single
`Bool`. `Matrix` therefore does not conform to `EqualityComparable` on purpose.
To ask whether two matrices are wholly identical, use
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

`+=`, `-=`, `*=`, `/=`, `//=` and `%=` are defined on `Matrix` only, and accept
a matrix, a view, or a scalar. Unlike the out-of-place operators they allocate
nothing: they write back through the matrix's own strides, so a transposed or
column-major matrix keeps its layout.

`MatrixView` has no in-place operators, for the same reason it has no `store`
method: the type is generic over its origin, and Mojo checks a method body
against the read-only instantiation too, so nothing that writes through
`self.data` can be defined on it. Mutate a view through the free functions in
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
var A = la.zeros[DType.float64](3, 3)

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

var B = la.zeros[DType.float64](4, 4)

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
Use `get_size()` for the element count.

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
| `sum`, `prod`       | `statistics`, `math` | `Scalar[dtype]`     | `Matrix[dtype]`                |
| `min`, `max`        | `math`               | `Scalar[dtype]`     | `Matrix[dtype]`                |
| `cumsum`, `cumprod` | `statistics`, `math` | same shape, scanned | same shape, scanned            |
| `argmin`, `argmax`  | `searching`          | `Int`, row-major    | `Matrix[DType.int64]`          |
| `all`, `any`        | `logic`              | `Bool`              | `Matrix[DType.bool]`           |
| `sort`              | `sorting`            | —                   | axis required                  |
| `argsort`           | `sorting`            | —                   | `Matrix[DType.int64]`          |
| `sort_inplace`      | `sorting`            | —                   | axis required, writes `Matrix` |

```mojo
from linamo.routines.statistics import sum, cumsum
from linamo.routines.math import min, max, prod
from linamo.routines.searching import argmax
from linamo.routines.logic import all, any
from linamo.routines.sorting import sort, sort_inplace

var A = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

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
    dtype: DType, origin: Origin[mut=False]
](v: la.MatrixView[dtype, origin]) -> Scalar[dtype]:
    var n = Scalar[dtype](0)
    for i in range(v.nrows):
        for j in range(v.ncols):
            if v[i, j] > 0:
                n += 1
    return n

var counts = apply_along_axis[
    axis=1, func=_count_positive[DType.float64, origin_of(A.data)]
](A.view())
```

`axis` is a compile-time parameter, so the traversal is specialised rather than
branched at run time, and the kernel is a compile-time parameter too, so it
inlines. The kernel's origin has to name the buffer it will be handed, which is
what `origin_of(A.data)` is doing there.

---

## Shape and layout

### The two layouts

A `Matrix` stores its elements row-major (C-contiguous) or column-major
(Fortran-contiguous), chosen with `order` at creation. The layout changes where
the elements sit in memory and nothing about what `A[i, j]` means.

Shape and layout are readable at any time:

| Query                      | Answer                                    |
| -------------------------- | ----------------------------------------- |
| `nrows`, `ncols`           | the shape                                 |
| `get_size()`               | `nrows * ncols`                           |
| `len(m)`                   | `nrows`, to agree with iteration          |
| `row_stride`, `col_stride` | the distance in memory between neighbours |
| `is_c_contiguous()`        | dense and row-major                       |
| `is_f_contiguous()`        | dense and column-major                    |
| `is_row_contiguous()`      | neighbours along a row are adjacent       |
| `is_col_contiguous()`      | neighbours down a column are adjacent     |

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
var A = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

var R = la.reshape(A, 3, 2)          # copies
var V = la.reshape_view(A.view(), 3, 2)  # no copy, shares A's buffer
var F = la.flatten(A)                # 1 x 6
var C = la.contiguous(A[0:2:1, 0:3:2], "C")  # densify a strided view
var I = la.astype[DType.int32](A)    # truncates towards zero
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

| Routine        | Returns              | Notes                                    |
| -------------- | -------------------- | ---------------------------------------- |
| `transpose(a)` | `Matrix`             | a new matrix, axes exchanged             |
| `trace(a)`     | `Scalar[dtype]`      | square input required                    |
| `det(a)`       | `Scalar[dtype]`      | via LU                                   |
| `inv(a)`       | `Matrix`             | via LU; solves `A @ X = I`               |
| `lu(a)`        | `(L, U, piv)`        | partial pivoting, `PA = LU`              |
| `cholesky(a)`  | `Matrix` (lower `L`) | symmetric positive-definite input        |
| `qr(a)`        | `(Q, R)`             | Householder reflections, `m >= n`        |
| `solve(A, b)`  | `Matrix`             | via LU; `b` may have several columns     |
| `lstsq(A, b)`  | `Matrix`             | via QR; overdetermined systems, `m >= n` |

```mojo
var A = la.matrix[DType.float64](
    [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
)
var b = la.matrix[DType.float64]([[1.0], [2.0], [3.0]])

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

A singular matrix raises `ValueError` from `solve`, `inv` and `det`; a
non-square input raises from the routines that need one; and `cholesky` raises
if the input is not positive definite. Nothing returns a silent NaN.

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
including the read-only one - so no method that writes through `self.data` can
even be defined. Bulk writes on views are therefore free functions in
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
looping over `self.data` itself, so each write has exactly one implementation
in the library.

Note that region assignment is spelled as a named method rather than
`m[a:b, c:d] = src`. Mojo routes slice-assignment through `__getitem__`, which
would force the right-hand side to be a view carrying the *target's* own
origin - making assignment from any other matrix inexpressible.

---

## NumPy interoperability

```mojo
from linamo.routines.numpy_interop import matrix_from_numpy, to_numpy
from std.python import Python

def main() raises:
    var np = Python.import_module("numpy")
    var arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    var A = matrix_from_numpy[DType.float64](arr)   # numpy -> Linamo
    var back = to_numpy(A)                          # Linamo -> numpy
```

Both directions **copy**. The input array must be 2-D and non-empty, and the
results are always C-contiguous regardless of the source's layout. The dtype is
not inferred from the array — name it as a parameter, and the conversion raises
if the array cannot supply it.

There is no `to_ndarray` for NuMojo. NuMojo's own `Matrix` type is gone, and
reproducing its N-dimensional array here would mean depending on it.

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

## StaticMatrix

`StaticMatrix[dtype, nrows, ncols]` carries its shape in its type and stores
its elements in a `SIMD` register buffer rather than on the heap. Nothing is
allocated, and the shape is known to the optimiser.

```mojo
var S = la.smatrix[2, 3, DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(S.get_size())        # 6
print(S[1, 2])             # 6.0
print(S.is_c_contiguous()) # False — see below
```

The buffer is padded to the next power of two in each dimension, so a `2 x 3`
matrix occupies a `2 x 4` SIMD vector and a `3 x 3` occupies `4 x 4`. That is
what makes the register storage possible, and it is why the type suits small
fixed-size matrices — the 2×2, 3×3 and 4×4 of geometry — rather than large
ones. It is also why `is_c_contiguous()` is False unless `ncols` is already a
power of two: the row stride is the padded width, not `ncols`.

`StaticMatrix` is the least developed of the three types. It has the shape and
layout queries, element reads, printing, and `+` and `@`. It does not have the
rest of the operator set, element writes, slicing, views, or any of the
routines. Use `Matrix` for anything beyond small fixed-size storage and those
two operations.

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
def add[dtype: DType, origin_a: Origin, origin_b: Origin](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises -> Matrix[dtype]:
    return _elementwise_view[func=Scalar[dtype].__add__](a, b)
```

`add(a, b)` still compiles when either operand is a `Matrix`, because
`MatrixView` carries an `@implicit` constructor from `Matrix` and the compiler
inserts the conversion. It is the same O(1) metadata copy `view()` performs, so
nothing is allocated and nothing is copied.

Two details make that constructor safe, and both are load-bearing:

```mojo
@implicit
def __init__[d: DType](
    out self: MatrixView[d, ImmOrigin(origin_of(m.data))], ref m: Matrix[d]
):
```

The argument is `ref m`, and only `ref` binds the origin to the *caller's*
storage. Under `imm`, `read` or the default convention the argument gets its
own origin --- `origin_of(m.data)` then names the callee's parameter slot ---
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
_elementwise_view[func=Scalar[dtype].__add__](a, b)
_elementwise_view[func=Scalar[dtype].__mul__](a, b)
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
