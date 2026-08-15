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
and `Matrix.__getitem__` take `self` by `ref`, so a view taken from a mutable
matrix is writable and a view taken from a borrowed one is read-only. Writing
through a read-only view is a compile error, not a runtime check.

## Mutability of indexing and slicing

| Expression      | Receiver     | Result       | Mutability  |
| --------------- | ------------ | ------------ | ----------- |
| `m[i, j]`       | `Matrix`     | reference    | follows `m` |
| `m[a:b, c:d]`   | `Matrix`     | `MatrixView` | follows `m` |
| `m.view()`      | `Matrix`     | `MatrixView` | follows `m` |
| `m.rows()`      | `Matrix`     | row views    | follows `m` |
| `m.cols()`      | `Matrix`     | column views | follows `m` |
| `v[i, j]`       | `MatrixView` | reference    | same as `v` |
| `v[a:b, c:d]`   | `MatrixView` | `MatrixView` | same as `v` |
| `v.to_matrix()` | `MatrixView` | `Matrix`     | new owner   |

"Follows `m`" means the borrow checker decides. A `var` matrix yields writable
views; a matrix received as an immutable argument yields read-only ones. The
same rule applies transitively, so a view of a view of a read-only matrix stays
read-only.

## Copying and assignment

`Matrix` is explicitly copyable but not *implicitly* copyable, so a plain
`var a = b` will not silently duplicate a buffer -- the compiler rejects it and
asks you to say what you meant. `MatrixView` is implicitly copyable, because
copying one only duplicates a span and five integers.

| Expression              | Result                            | Cost   |
| ----------------------- | --------------------------------- | ------ |
| `var a = b`             | compile error, suggests `.copy()` | --     |
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
including the read-only one -- so no method that writes through `self.data` can
even be defined. Bulk writes on views are therefore free functions in
`routines/mutation.mojo`, pinned to a mutable origin:

- `store[width](view, row, col, value)` -- write a SIMD run.
- `fill(view, rows, cols, value)` -- write one scalar across a region.
- `assign(view, rows, cols, src)` -- copy a block into a region.

Passing a read-only view to any of them fails to compile at the call site,
which is the guarantee we wanted. Single-element writes need none of this:
`v[i, j] = x` writes through the reference returned by `__getitem__`, where the
caller's origin is already concrete.

`Matrix` carries `fill` and `assign` as methods too, for the common case where
you are working with an owned matrix directly.

Note that region assignment is spelled as a named method rather than
`m[a:b, c:d] = src`. Mojo routes slice-assignment through `__getitem__`, which
would force the right-hand side to be a view carrying the *target's* own
origin -- making assignment from any other matrix inexpressible.

## Inter-operability of Matrix and MatrixView

The `Matrix` and `MatrixView` classes are designed to inter-operate seamlessly.
You can create a `MatrixView` from a `Matrix`, and you can also create a new
`Matrix` from a `MatrixView`. This allows for flexible manipulation of matrix
data without unnecessary copying, while still maintaining clear ownership
semantics.

If an operation is conducted between a `Matrix` and a `MatrixView`, the result
is typically a new `Matrix` that owns its data.

The operations can be defined using generic functions that accept both `Matrix`
and `MatrixView` types, allowing for polymorphic behavior while ensuring that
the underlying data is handled correctly based on ownership and mutability
rules.

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
