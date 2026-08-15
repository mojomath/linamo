# API

## Matrix class

The `Matrix` class owns its data and can write to it. The elements are stored in
a contiguous block of memory in either row-major (C-contiguous) or column-major
(Fortran-contiguous) order.

## MatrixView class

The `MatrixView` class is a non-owning view of a matrix. It can be used to
create submatrices or to view the same data with different offsets, shapes, and
strides. A `MatrixView` cannot write to the data it views, and it does not
manage the memory of the data.

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
