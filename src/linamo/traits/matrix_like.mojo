trait MatrixLike(Copyable):
    """A trait for types that behave like matrices, providing a common interface
    for matrix operations.
    """

    def get_nrows(self) -> Int:
        """Returns the number of rows in the matrix-like object."""
        ...

    def get_ncols(self) -> Int:
        """Returns the number of columns in the matrix-like object."""
        ...

    def get_row_stride(self) -> Int:
        """Returns the row stride of the matrix-like object."""
        ...

    def get_col_stride(self) -> Int:
        """Returns the column stride of the matrix-like object."""
        ...

    def get_offset(self) -> Int:
        """Returns the offset in the underlying data buffer for the matrix-like
        object."""
        ...

    def get_size(self) -> Int:
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
