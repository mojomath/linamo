"""
This module provides bulk write operations on mutable matrix views.

These are free functions rather than `MatrixView` methods for a concrete
reason. `MatrixView` is generic over `origin: Origin[mut=mut]`, and Mojo 1.0
type-checks a method body against *every* instantiation, including the
read-only one -- so any method that writes through `self.data` is rejected at
definition time, and neither a `where Self.mut` clause nor a constrained `self`
refines it. Pinning these functions to `Origin[mut=True]` moves the requirement
into the signature instead: passing a read-only view is a compile error at the
call site, which is the guarantee we wanted anyway.

Single-element writes do not need any of this -- `view[i, j] = x` writes
through the reference returned by `MatrixView.__getitem__`, where the caller's
origin is concrete.
"""

from matmojo.types.errors import IndexError, ValueError
from matmojo.types.matrix import Matrix
from matmojo.types.matrix_view import MatrixView
from matmojo.utils.indexing import get_offset, indices_within_bounds


def store[
    dtype: DType, origin: Origin[mut=True], //, width: Int = 1
](
    view: MatrixView[dtype, origin],
    row: Int,
    col: Int,
    value: SIMD[dtype, width],
) raises:
    """Stores `width` elements along row `row`, starting at column `col`.

    Mirrors `MatrixView.load`: the contiguous case walks consecutive addresses
    so the loop vectorises, and the strided case scatters. Both are correct;
    only the speed differs.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the target view, which must be mutable.
        width: How many elements to store.

    Args:
        view: The view to write into.
        row: The row to write to.
        col: The column at which the run starts.
        value: The elements to write.

    Raises:
        IndexError: If the run would leave the view.
    """
    if row < 0 or row >= view.nrows or col < 0 or col + width > view.ncols:
        raise IndexError(
            file="src/matmojo/routines/mutation.mojo",
            function="store[width](view, row: Int, col: Int, value)",
            message="SIMD store runs past the end of the view.",
            previous_error=None,
        )
    var base = get_offset(
        row, col, view.row_stride, view.col_stride, view.offset
    )
    if view.col_stride == 1:
        for i in range(width):
            view.data[base + i] = value[i]
    else:
        for i in range(width):
            view.data[base + i * view.col_stride] = value[i]


def fill[
    dtype: DType, origin: Origin[mut=True], //
](
    view: MatrixView[dtype, origin],
    rows: Slice,
    cols: Slice,
    value: Scalar[dtype],
) raises:
    """Writes one scalar into every element of the selected sub-view.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the target view, which must be mutable.

    Args:
        view: The view to write into.
        rows: The rows to fill.
        cols: The columns to fill.
        value: The scalar written to every selected element.
    """
    var target = view[rows, cols]
    for i in range(target.nrows):
        for j in range(target.ncols):
            target[i, j] = value


def assign[
    dtype: DType,
    origin: Origin[mut=True],
    mut_b: Bool,
    //,
    origin_b: Origin[mut=mut_b],
](
    view: MatrixView[dtype, origin],
    rows: Slice,
    cols: Slice,
    src: MatrixView[dtype, origin_b],
) raises:
    """Copies `src` into the sub-view selected by `rows` and `cols`.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the target view, which must be mutable.
        mut_b: Whether the source view is mutable.
        origin_b: The origin of the source view.

    Args:
        view: The view to write into.
        rows: The rows to assign into.
        cols: The columns to assign into.
        src: The source, which must match the target shape exactly.

    Raises:
        ValueError: If the shapes do not match.
    """
    var target = view[rows, cols]
    if target.nrows != src.nrows or target.ncols != src.ncols:
        raise ValueError(
            file="src/matmojo/routines/mutation.mojo",
            function="assign(view, rows, cols, src)",
            message="Shape mismatch in region assignment.",
            previous_error=None,
        )
    for i in range(target.nrows):
        for j in range(target.ncols):
            target[i, j] = src[i, j]


def assign[
    dtype: DType, origin: Origin[mut=True], //
](
    view: MatrixView[dtype, origin],
    rows: Slice,
    cols: Slice,
    src: Matrix[dtype],
) raises:
    """Copies a `Matrix` into the sub-view selected by `rows` and `cols`.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the target view, which must be mutable.

    Args:
        view: The view to write into.
        rows: The rows to assign into.
        cols: The columns to assign into.
        src: The source, which must match the target shape exactly.

    Raises:
        ValueError: If the shapes do not match.
    """
    assign(view, rows, cols, src.view())
