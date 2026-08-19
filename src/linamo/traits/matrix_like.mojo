"""A common interface for the shape and stride accessors of a matrix type.

No type in the library conforms to this trait and nothing is generic over it.
`Matrix`, `MatrixView` and `StaticMatrix` declare the methods below as ordinary
methods instead.

Operand genericity does not go through here: a routine that accepts either a
`Matrix` or a `MatrixView` gets that from the `@implicit` constructor on
`MatrixView`, which a trait method cannot express, because the converted type's
`origin` parameter depends on the borrow of the argument. See the note in
`linamo.routines.math`.

What a trait could carry is the read-only algorithms that both types spell out
separately today, `__str__` and `write_to` foremost. Mojo 1.0 supports
associated aliases, so `comptime dtype: DType` alongside
`def at(self, r: Int, c: Int) -> Scalar[Self.dtype]` is expressible, and that
is the shape such a trait would take.
"""


trait MatrixLike(Copyable):
    """A common interface for matrix-like types. Currently unused."""

    def nrows(self) -> Int:
        """Returns the number of rows in the matrix-like object."""
        ...

    def ncols(self) -> Int:
        """Returns the number of columns in the matrix-like object."""
        ...

    def row_stride(self) -> Int:
        """Returns the row stride of the matrix-like object."""
        ...

    def col_stride(self) -> Int:
        """Returns the column stride of the matrix-like object."""
        ...

    def offset(self) -> Int:
        """Returns the offset in the underlying data buffer for the matrix-like
        object."""
        ...

    def size(self) -> Int:
        """Returns the total number of elements in the matrix-like object."""
        ...

    def is_c_contiguous(self) -> Bool:
        """Returns True if the data is stored in row-major (C) contiguous order.

        A matrix is C-contiguous when `col_stride == 1` and
        `row_stride == ncols`, meaning elements within a row are adjacent
        in memory.
        """
        ...

    def is_f_contiguous(self) -> Bool:
        """Returns True if the data is stored in column-major (Fortran) contiguous order.

        A matrix is F-contiguous when `row_stride == 1` and
        `col_stride == nrows`, meaning elements within a column are adjacent
        in memory.
        """
        ...

    def is_row_contiguous(self) -> Bool:
        """Returns True if elements within each row are contiguous in memory.

        This requires `col_stride == 1`. Unlike `is_c_contiguous()`, this
        allows padding between rows (row_stride >= ncols).  Many SIMD kernels
        (e.g. matmul) only need this weaker guarantee.
        """
        ...

    def is_col_contiguous(self) -> Bool:
        """Returns True if elements within each column are contiguous in memory.

        This requires `row_stride == 1`. Unlike `is_f_contiguous()`, this
        allows padding between columns (col_stride >= nrows).  Many SIMD kernels
        only need this weaker guarantee.
        """
        ...

    def copy(self) -> Self:
        """Returns a copy of the matrix-like object."""
        ...

    def __str__(self) -> String:
        """Returns a string representation of the matrix-like object."""
        ...

    # fn __getitem__(self, row: Int, col: Int) -> Scalar:
    #     """Returns the element at the specified row and column indices."""
    #     ...
