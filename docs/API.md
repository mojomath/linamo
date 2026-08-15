# API

## Matrix class

The `Matrix` class owns its data and can write to it. The elements are stored in
a contiguous block of memory in either row-major (C-contiguous) or column-major
(Fortran-contiguous) order.

## MatrixView class

The `MatrixView` class is a non-owning view of a matrix. It can be used to
create submatrices or to view the same data with different offsets, shapes, and
strides. It does not manage the memory of the data it points at.

Whether a view can *write* to that data is not a property of `MatrixView`
itself: it is inherited from whatever the view was made from. `Matrix.view()`
takes `self` by `ref`, so a view taken from a mutable matrix is writable and a
view taken from a borrowed one is read-only. Slicing with `m[a:b, c:d]` is the
deliberate exception and is always read-only.
Writing through a read-only view is a compile error, not a runtime check.

## Mutability of indexing and slicing

The rule is one sentence:

> **Nothing that carries a borrow in its type is ever handed out mutable,
> except through a function in `linamo.routines.mutation`.**

A `MatrixView` carries a borrow in its type (that is what its `origin`
parameter is) so every view produced by a method is read-only, no matter how
the receiver was bound. Writing is the rarer case and gets the explicit
spelling.

### From a `Matrix`

| Expression            | Result       | Receiver is `var` | Receiver is read-only | Rule     |
| --------------------- | ------------ | ----------------- | --------------------- | -------- |
| `m[i, j]`             | reference    | **mutable**       | read-only             | inherits |
| `m[a:b, c:d]`         | `MatrixView` | read-only         | read-only             | fixed    |
| `m.view()`            | `MatrixView` | read-only         | read-only             | fixed    |
| `m.rows()`/`m.cols()` | views        | read-only         | read-only             | fixed    |
| `for r in m`          | views        | read-only         | read-only             | fixed    |

`m[i, j]` is the one place a method still propagates the caller's mutability,
and it is the mutation you want on an owner: `m[i, j] = x` writes an element.
It is safe to inherit because a bare reference is consumed where it is formed,
unlike a view, which is a value you can bind, store and pass twice.

`m.view()` is exactly `m[:, :]`, and exists because the named routines accept a
`MatrixView` - it is the O(1) conversion that lets `add(a, b)` reach the same
kernel whether its arguments are matrices or views.

### From a `MatrixView`

A view's own mutability is carried in its type, as the `mut` field of its
`origin` parameter - not as a runtime flag. A view is mutable only if it came
from `linamo.routines.mutation`.

| Expression            | Result       | Receiver mutable | Receiver read-only | Rule     |
| --------------------- | ------------ | ---------------- | ------------------ | -------- |
| `v[i, j]`             | reference    | **mutable**      | read-only          | inherits |
| `v[a:b, c:d]`         | `MatrixView` | read-only        | read-only          | fixed    |
| `v.rows()`/`v.cols()` | views        | **mutable**      | read-only          | inherits |
| `v.as_imm()`          | `MatrixView` | read-only        | read-only          | fixed    |
| `v.to_matrix()`       | `Matrix`     | new owner        | new owner          | copies   |

Every *fixed* row is one-way: nothing turns a read-only view back into a
mutable one. The *inherits* rows on a view are harmless precisely because a
mutable view can only have come from the mutation module in the first place.

### Writing

| You want to          | Write                                        |
| -------------------- | -------------------------------------------- |
| write one element    | `m[i, j] = x`                                |
| write a region       | `assign(view_mut(m, x, y), rows, cols, src)` |
| fill a region        | `fill(view_mut(m, x, y), rows, cols, value)` |
| write rows in a loop | `for row in rows_mut(m): ...`                |
| get a writable view  | `view_mut(m, x, y)`                          |

All of these except the first come from `linamo.routines.mutation`. A caller
who never imports that module cannot construct a mutable view at all.

### Which conversions exist

| From \ To              | `Matrix`        | `MatrixView` read-only | `MatrixView` mutable | element ref read-only | element ref mutable |
| ---------------------- | --------------- | ---------------------- | -------------------- | --------------------- | ------------------- |
| `var Matrix`           | `m.copy()`      | `m[a:b, c:d]`          | `view_mut(m, x, y)`  | -                     | `m[i, j]`           |
| read-only `Matrix`     | `m.copy()`      | `m.view()`             | **impossible**       | `m[i, j]`             | **impossible**      |
| read-only `MatrixView` | `v.to_matrix()` | `v[a:b, c:d]`          | **impossible**       | `v[i, j]`             | **impossible**      |
| mutable `MatrixView`   | `v.to_matrix()` | `v.as_imm()`           | `view_mut(v, x, y)`  | `v.as_imm()[i, j]`    | `v[i, j]`           |

The mutable column has exactly one spelling, and the two **impossible** columns
are the invariant the design rests on: mutability is only ever lost, never
gained. Nothing in the library promotes a read-only value to a writable one.

The audit is mechanical. A method can only hand back the caller's mutability by
taking `ref self`, so:

```console
$ grep -rn "ref self" src/linamo/types/
src/linamo/types/matrix.mojo:208:        ref self, row: Int, col: Int
```

One line, element access. Anything else appearing there is a hole.

### Two things that are deliberately absent

`m[a:b, c:d] = src` is not available. Defining `__setitem__` on `Matrix` makes
the compiler pass `self` to `__getitem__` as a temporary copy in some
positions, so a sliced view ends up carrying the origin of a dead temporary and
`a[0:1, :] - a[1:2, :]` stops compiling. Region assignment is `assign(...)`.

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

## Iteration

`len()` returns the number of rows, so it agrees with what iteration yields.
Use `get_size()` for the element count.

Iterating a matrix or a view walks its rows, yielding each as a `1 x ncols`
view onto the parent buffer. Nothing is copied, and each row inherits the
parent's mutability, so rows of a mutable matrix can be written through.
`cols()` does the same by column, and both take a `forward` parameter, so
`rows[False]()` walks bottom to top.

Mojo's builtin `reversed()` only accepts specific standard-library containers,
so it will not dispatch to `__reversed__` on these types. Call `rows[False]()`,
which is clearer at the call site anyway.

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
- `fill(view, rows, cols, value)` - write one scalar across a region.
- `assign(view, rows, cols, src)` - copy a block into a region.

Passing a read-only view to any of them fails to compile at the call site,
which is the guarantee we wanted. Single-element writes need none of this:
`v[i, j] = x` writes through the reference returned by `__getitem__`, where the
caller's origin is already concrete.

`Matrix` carries `fill` and `assign` as methods too, for the common case where
you are working with an owned matrix directly.

Note that region assignment is spelled as a named method rather than
`m[a:b, c:d] = src`. Mojo routes slice-assignment through `__getitem__`, which
would force the right-hand side to be a view carrying the *target's* own
origin - making assignment from any other matrix inexpressible.

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

### Comparisons return masks

`a == b` is an element-wise `Matrix[DType.bool]` of the same shape, not a single
`Bool`. `Matrix` therefore does not conform to `EqualityComparable` on purpose.
To ask whether two matrices are wholly identical, use
`assert_matrices_equal` / `assert_matrices_close` from `utils/test_utils.mojo`.

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

## Inter-operability of Matrix and MatrixView

`Matrix` and `MatrixView` inter-operate freely. Any binary routine or operator
accepts either type on either side, and the result is always a **new** `Matrix`
that owns its data.

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

**The permutations collapse at the routine layer.** `Matrix` becomes
`MatrixView` through `view()`, which is O(1) and allocates nothing, so the four
overloads of a routine are one line each and all end in the same call:

```mojo
def add[dtype: DType, origin_a: Origin, origin_b: Origin](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises -> Matrix[dtype]:
    return _elementwise_view[func=Scalar[dtype].__add__](a, b)

def add[dtype: DType](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[dtype]:
    return _elementwise_view[func=Scalar[dtype].__add__](a.view(), b.view())

# ... and the two mixed pairs, likewise
```

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

## Order of memory layout

The `Matrix` class supports both row-major (C-contiguous) and column-major
(Fortran-contiguous) memory layouts. The order of memory layout can be specified
when creating a `Matrix`, and it determines how the elements are stored in
memory.

To optimize the performance of matrix operations, some functions may be
implemented in several versions that are optimized for different memory layouts.
For example, matrix multiplication may have separate implementations for:

1. c@c: Both matrices are row-major.
1. f@f: Both matrices are column-major.
1. c@f, f@c: A row-major matrix multiplied by a column-major matrix, and vice
   versa.
1. c@v, f@v, v@c, v@f: A matrix multiplied by a non-contiguous view. This may
   require a naive implementation that does not assume any specific memory
   layout for the view.
