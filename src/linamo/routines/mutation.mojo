"""
This module is the only place in the library that hands out a mutable view.

Everything on `Matrix` and `MatrixView` is read-only: slicing, `view()`,
`rows()`, `cols()` and iteration all yield views that cannot be written
through, no matter how the receiver was bound. That is deliberate. A mutable
view is an *exclusive* borrow, and Mojo refuses to pass two values borrowing
the same memory into one call, so if slicing inherited mutability then
`a[0:1, :] - a[1:2, :]` would not compile. Reading a matrix twice at once is
always safe; writing is the rarer case, and it is the one that gets the
explicit spelling.

The invariant, which is worth stating because it can be checked mechanically:

    A `MatrixView` is mutable if and only if it came from a function in this
    module. `grep "ref self" src/` returns exactly one line - element access
    on `Matrix` - and nothing else in the library can propagate write access.

So a caller who never imports from `linamo.routines.mutation` cannot construct
a mutable view at all.

This module provides bulk write operations on mutable matrix views.

These are free functions rather than methods for a concrete reason. 
`MatrixView` is generic over `origin: Origin[mut=mut]`, and Mojo 1.0
type-checks a method body against *every* instantiation, including the
read-only one - so any method that writes through `self.data` is rejected at
definition time, and neither a `where Self.mut` clause nor a constrained `self`
refines it. Pinning these functions to `Origin[mut=True]` moves the requirement
into the signature instead: passing a read-only view is a compile error at the
call site, which is the guarantee we wanted anyway.

Single-element writes do not need any of this - `view[i, j] = x` writes
through the reference returned by `MatrixView.__getitem__`, where the caller's
origin is concrete. Likewise `m[i, j] = x` writes straight through the owner.

Note that region assignment cannot be spelled `m[a:b, c:d] = src`. Defining
`__setitem__` on `Matrix` makes the compiler pass `self` to `__getitem__` as a
temporary copy in some positions, so a sliced view ends up carrying the origin
of a dead temporary and ordinary expressions like `a[0:1, :] - a[1:2, :]` stop
compiling. `assign()` below is the region write.
"""

from linamo.types.errors import IndexError, ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_iter import MatrixAxisIter
from linamo.types.matrix_view import MatrixView
from linamo.utils.indexing import get_offset, indices_within_bounds


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
            function="store[width](view, row: Int, col: Int, value)",
            message="SIMD store runs past the end of the view.",
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
    var target = view_mut(view, rows, cols)
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
    var target = view_mut(view, rows, cols)
    if target.nrows != src.nrows or target.ncols != src.ncols:
        raise ValueError(
            function="assign(view, rows, cols, src)",
            message="Shape mismatch in region assignment.",
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


# ===----------------------------------------------------------------------===#
# Mutable views
# ===----------------------------------------------------------------------===#


def view_mut[
    dtype: DType
](ref m: Matrix[dtype], x: Slice, y: Slice) raises -> MatrixView[
    dtype, origin_of(m.data)
]:
    """Gets a writable view of a region of a matrix.

    This is the counterpart of `m[x, y]`, which is always read-only. The view
    returned here inherits the mutability of `m`, so it is writable when `m` is
    a `var` and read-only otherwise - and passing a read-only one to `fill`,
    `store` or `assign` is a compile error rather than a runtime check.

    A mutable view is an exclusive borrow, so it cannot appear twice in one
    expression. Use `MatrixView.as_imm()` to demote it when it has to be
    combined with another view of the same matrix.

    Args:
        m: The matrix to view.
        x: The rows to include.
        y: The columns to include.

    Returns:
        A view of the region, writable exactly when `m` is.
    """
    return MatrixView(
        data=Span(m.data),
        slice_x=x,
        slice_y=y,
        initial_nrows=m.nrows,
        initial_ncols=m.ncols,
        initial_row_stride=m.row_stride,
        initial_col_stride=m.col_stride,
        initial_offset=0,
    )


def view_mut[
    dtype: DType, origin: Origin[mut=True], //
](view: MatrixView[dtype, origin], x: Slice, y: Slice) raises -> MatrixView[
    dtype, origin
]:
    """Gets a writable sub-view of an already-writable view.

    Unlike `v[x, y]`, which demotes to a read-only view, this keeps the parent
    view's origin. Pinning the parameter to `Origin[mut=True]` means a
    read-only view is rejected at the call site.

    Args:
        view: The view to sub-view.
        x: The rows to include.
        y: The columns to include.

    Returns:
        A writable view of the region.
    """
    return MatrixView[dtype, origin](
        data=view.data,
        slice_x=x,
        slice_y=y,
        initial_nrows=view.nrows,
        initial_ncols=view.ncols,
        initial_row_stride=view.row_stride,
        initial_col_stride=view.col_stride,
        initial_offset=view.offset,
    )


def rows_mut[
    dtype: DType, //, forward: Bool = True
](ref m: Matrix[dtype]) -> MatrixAxisIter[dtype, origin_of(m.data), 0, forward]:
    """Walks the rows of a matrix, yielding each as a writable view.

    `m.rows()` is read-only because iteration is an implicit path. This is the
    explicit one, and it is what an in-place row operation wants.

    Parameters:
        forward: True for first-to-last, False for last-to-first.

    Args:
        m: The matrix to walk.

    Returns:
        An iterator yielding writable `1 x ncols` views.
    """
    return MatrixAxisIter[axis=0, forward=forward](
        MatrixView(
            data=Span(m.data),
            nrows=m.nrows,
            ncols=m.ncols,
            row_stride=m.row_stride,
            col_stride=m.col_stride,
            offset=0,
        )
    )


def cols_mut[
    dtype: DType, //, forward: Bool = True
](ref m: Matrix[dtype]) -> MatrixAxisIter[dtype, origin_of(m.data), 1, forward]:
    """Walks the columns of a matrix, yielding each as a writable view.

    Parameters:
        forward: True for first-to-last, False for last-to-first.

    Args:
        m: The matrix to walk.

    Returns:
        An iterator yielding writable `nrows x 1` views.
    """
    return MatrixAxisIter[axis=1, forward=forward](
        MatrixView(
            data=Span(m.data),
            nrows=m.nrows,
            ncols=m.ncols,
            row_stride=m.row_stride,
            col_stride=m.col_stride,
            offset=0,
        )
    )
