"""This module provides indexing and memory layout utilities for Linamo."""


@always_inline
def get_offset(
    row: Int, col: Int, row_stride: Int, col_stride: Int, initial_offset: Int
) -> Int:
    """Calculates the linear offset in a buffer for given indices.

    This function computes the position of an element in a 1D buffer
    based on its 2D indices, strides, and an optional initial offset.

    Args:
        row: The row index of the element to access.
        col: The column index of the element to access.
        row_stride: The stride for the row dimension.
        col_stride: The stride for the column dimension.
        initial_offset: An optional offset to add to the computed position.
            Defaults to 0. Useful for views or slices.

    Returns:
        The linear offset in the buffer where the element is located.

    Examples:
        # For a matrix with row-major layout (row_stride=4, col_stride=1):
        get_offset(2, 3, 4, 1, 1)  # Returns 12 (2*4 + 3*1 + 1).

        # For a matrix with column-major layout (row_stride=1, col_stride=4):
        get_offset(2, 3, 1, 4, 0)  # Returns 14 (2*1 + 3*4).
    """
    return initial_offset + row * row_stride + col * col_stride


@always_inline
def get_offset(row: Int, col: Int, row_stride: Int, col_stride: Int) -> Int:
    """Calculates the linear offset in a buffer for given indices.

    This function computes the position of an element in a 1D buffer
    based on its 2D indices, strides, and an optional initial offset.

    Args:
        row: The row index of the element to access.
        col: The column index of the element to access.
        row_stride: The stride for the row dimension.
        col_stride: The stride for the column dimension.

    Returns:
        The linear offset in the buffer where the element is located.

    Examples:
        # For a matrix with row-major layout (row_stride=4, col_stride=1):
        get_offset(2, 3, 4, 1)  # Returns 11 (2*4 + 3*1).

        # For a matrix with column-major layout (row_stride=1, col_stride=4):
        get_offset(2, 3, 1, 4)  # Returns 14 (2*1 + 3*4).
    """
    return row * row_stride + col * col_stride


@always_inline
def indices_within_bounds(row: Int, col: Int, nrows: Int, ncols: Int) -> Bool:
    """Checks if the given row and column indices are within bounds."""
    return (row >= 0) and (row < nrows) and (col >= 0) and (col < ncols)


@always_inline
def indices_out_of_bounds(row: Int, col: Int, nrows: Int, ncols: Int) -> Bool:
    """Checks if the given row and column indices are out of bounds."""
    return (row < 0) or (row >= nrows) or (col < 0) or (col >= ncols)


@always_inline
def layout_fits_buffer(
    nrows: Int, ncols: Int, row_stride: Int, col_stride: Int, length: Int
) -> Bool:
    """Checks that every index of a matrix lands inside a buffer.

    The largest offset any index reaches is the one at `[nrows - 1,
    ncols - 1]`, so bounding that bounds them all. An empty matrix reaches no
    offset at all and always fits.

    Args:
        nrows: The number of rows.
        ncols: The number of columns.
        row_stride: The stride for the row dimension.
        col_stride: The stride for the column dimension.
        length: The number of elements in the buffer.

    Returns:
        True if every index is in range.
    """
    if nrows == 0 or ncols == 0:
        return True
    return (nrows - 1) * row_stride + (ncols - 1) * col_stride < length


@always_inline
def layout_is_dense(
    nrows: Int, ncols: Int, row_stride: Int, col_stride: Int
) -> Bool:
    """Checks that a stride pair maps distinct indices to distinct offsets.

    Positive strides are necessary but not sufficient: `(1, 1)` on a 2x2 sends
    both `[0, 1]` and `[1, 0]` to offset 1. What rules that out is the stride
    pair being C-major (`row_stride == ncols * col_stride`) or F-major
    (`col_stride == nrows * row_stride`), which are the only two layouts an
    owning matrix is built with. A zero stride fails the positivity test; it is
    a legitimate state for a `MatrixView`, where `broadcast_to` produces one,
    and never for a matrix that owns its buffer.

    Args:
        nrows: The number of rows.
        ncols: The number of columns.
        row_stride: The stride for the row dimension.
        col_stride: The stride for the column dimension.

    Returns:
        True if the layout is C-major or F-major with no padding.
    """
    if nrows == 0 or ncols == 0:
        return True
    if row_stride <= 0 or col_stride <= 0:
        return False
    return row_stride == ncols * col_stride or col_stride == nrows * row_stride
