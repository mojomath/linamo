"""
Tests for __len__, row/column iteration, SIMD access and region assignment
on Matrix and MatrixView.
"""

import std.testing as testing
import linamo as la
from linamo.routines.mutation import (
    assign,
    fill,
    rows_mut,
    store,
    view_mut,
)


def test_len_is_row_count() raises:
    """`len()` reports rows, so it agrees with what iteration yields."""
    var mat = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )
    testing.assert_equal(len(mat), 3)
    testing.assert_equal(len(mat.view()), 3)
    # get_size() remains the element count.
    testing.assert_equal(mat.get_size(), 12)


def test_row_iteration_yields_views() raises:
    """Each row arrives as a 1 x ncols view onto the parent buffer."""
    var mat = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var seen = List[Float64]()
    var count = 0
    for row in mat:
        testing.assert_equal(row.nrows, 1)
        testing.assert_equal(row.ncols, 3)
        seen.append(row[0, 0])
        count += 1
    testing.assert_equal(count, 3)
    testing.assert_equal(seen[0], 1.0)
    testing.assert_equal(seen[1], 4.0)
    testing.assert_equal(seen[2], 7.0)


def test_reversed_row_iteration() raises:
    """Reverse iteration walks the rows last to first."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var seen = List[Float64]()
    for row in mat.rows[False]():
        seen.append(row[0, 0])
    testing.assert_equal(len(seen), 3)
    testing.assert_equal(seen[0], 5.0)
    testing.assert_equal(seen[1], 3.0)
    testing.assert_equal(seen[2], 1.0)


def test_column_iteration() raises:
    """Column iteration yields nrows x 1 views."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var seen = List[Float64]()
    for col in mat.cols():
        testing.assert_equal(col.nrows, 2)
        testing.assert_equal(col.ncols, 1)
        seen.append(col[1, 0])
    testing.assert_equal(len(seen), 3)
    testing.assert_equal(seen[0], 4.0)
    testing.assert_equal(seen[1], 5.0)
    testing.assert_equal(seen[2], 6.0)


def test_iteration_over_view_is_zero_copy() raises:
    """Writing through a row from `rows_mut` reaches the parent matrix.

    Plain iteration (`for row in mat`) yields read-only rows, so the writable
    walk has to be asked for by name.
    """
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    for row in rows_mut(mat):
        row[0, 0] = 99.0
    testing.assert_equal(mat[0, 0], 99.0)
    testing.assert_equal(mat[1, 0], 99.0)


def test_load_contiguous_and_strided() raises:
    """`load` agrees on contiguous rows and on strided views."""
    var mat = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )
    var contiguous = mat.load[4](1, 0)
    testing.assert_equal(contiguous[0], 5.0)
    testing.assert_equal(contiguous[3], 8.0)

    # Every other column: col_stride becomes 2, so this takes the gather path.
    var strided = mat[0:2, 0:4:2]
    var gathered = strided.load[2](1, 0)
    testing.assert_equal(gathered[0], 5.0)
    testing.assert_equal(gathered[1], 7.0)


def test_load_out_of_bounds_raises() raises:
    """A run that would leave the matrix raises rather than reading past it."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    with testing.assert_raises():
        _ = mat.load[4](0, 0)


def test_matrix_store() raises:
    """`store` writes a SIMD run back into the matrix.

    SIMD widths must be powers of two, so runs are sized 1, 2, 4, ...
    """
    var mat = la.matrix[DType.float64]([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mat.store[2](1, 0, SIMD[DType.float64, 2](7.0, 8.0))
    testing.assert_equal(mat[1, 0], 7.0)
    testing.assert_equal(mat[1, 1], 8.0)
    testing.assert_equal(mat[1, 2], 0.0)


def test_view_store_contiguous_and_strided() raises:
    """The view `store` routine handles both layouts."""
    var mat = la.matrix[DType.float64](
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    store[width=2](
        view_mut(mat, Slice(0, 2), Slice(0, 4)),
        0,
        0,
        SIMD[DType.float64, 2](1.0, 2.0),
    )
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 1], 2.0)

    var strided = view_mut(mat, Slice(0, 2), Slice(0, 4, 2))
    store[width=2](strided, 1, 0, SIMD[DType.float64, 2](5.0, 6.0))
    testing.assert_equal(mat[1, 0], 5.0)
    testing.assert_equal(mat[1, 2], 6.0)


def test_set_region_scalar() raises:
    """`set` writes one scalar across the selected region only."""
    var mat = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    mat.set(Slice(0, 2), Slice(1, 3), Float64(0.0))
    testing.assert_equal(mat[0, 1], 0.0)
    testing.assert_equal(mat[1, 2], 0.0)
    # Outside the region is untouched.
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[2, 2], 9.0)


def test_fill_region_through_view() raises:
    """The view `fill` routine writes through to the owner."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    fill(
        view_mut(mat, Slice(0, 2), Slice(0, 2)),
        Slice(0, 1),
        Slice(0, 2),
        Float64(-1.0),
    )
    testing.assert_equal(mat[0, 0], -1.0)
    testing.assert_equal(mat[0, 1], -1.0)
    testing.assert_equal(mat[1, 0], 3.0)


def whole_view_fill_needs_no_try[
    o: Origin[mut=True], //
](v: la.MatrixView[DType.float64, o]):
    """`fill(v, value)` is callable from a non-raising context.

    This `def` carries no `raises`, so it would not compile if the whole-view
    scalar write could fail. That is the assertion; the body is incidental.
    """
    fill(v, 5.0)


def test_fill_whole_view_does_not_raise() raises:
    """The whole-view scalar write is total, so it is declared non-raising."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    whole_view_fill_needs_no_try(mat.view_mut(Slice(0, 1), Slice(0, 2)))
    testing.assert_equal(mat[0, 0], 5.0)
    testing.assert_equal(mat[0, 1], 5.0)
    testing.assert_equal(mat[1, 0], 3.0)


def test_fill_whole_view_covers_strided() raises:
    """The whole-view write visits every element of a strided view."""
    var mat = la.matrix[DType.float64](
        [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    )
    fill(view_mut(mat, Slice(0, 2), Slice(0, 4, 2)), Float64(9.0))
    testing.assert_equal(mat[0, 0], 9.0)
    testing.assert_equal(mat[0, 2], 9.0)
    testing.assert_equal(mat[1, 0], 9.0)
    testing.assert_equal(mat[1, 2], 9.0)
    # The columns the view skips are untouched.
    testing.assert_equal(mat[0, 1], 1.0)
    testing.assert_equal(mat[1, 3], 1.0)


def test_set_region_source() raises:
    """`set` copies a source block into the selected region."""
    var mat = la.matrix[DType.float64](
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    var src = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    mat.set(Slice(1, 3), Slice(0, 2), src.view())
    testing.assert_equal(mat[1, 0], 1.0)
    testing.assert_equal(mat[1, 1], 2.0)
    testing.assert_equal(mat[2, 0], 3.0)
    testing.assert_equal(mat[2, 1], 4.0)
    testing.assert_equal(mat[0, 0], 0.0)


def test_set_shape_mismatch_raises() raises:
    """A mismatched source shape is rejected."""
    var mat = la.matrix[DType.float64]([[0.0, 0.0], [0.0, 0.0]])
    var src = la.matrix[DType.float64]([[1.0, 2.0, 3.0]])
    with testing.assert_raises():
        mat.set(Slice(0, 1), Slice(0, 2), src.view())


def test_assign_into_view() raises:
    """The view `assign` routine writes through to the owner."""
    var mat = la.matrix[DType.float64]([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    var src = la.matrix[DType.float64]([[5.0, 6.0]])
    assign(
        view_mut(mat, Slice(0, 2), Slice(0, 3)),
        Slice(0, 1),
        Slice(1, 3),
        src,
    )
    testing.assert_equal(mat[0, 1], 5.0)
    testing.assert_equal(mat[0, 2], 6.0)
    testing.assert_equal(mat[0, 0], 0.0)


def test_assign_whole_view() raises:
    """`assign(v, src)` copies into every element of the view."""
    var mat = la.matrix[DType.float64]([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    var src = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    assign(view_mut(mat, Slice(0, 2), Slice(1, 3)), src)
    testing.assert_equal(mat[0, 1], 1.0)
    testing.assert_equal(mat[0, 2], 2.0)
    testing.assert_equal(mat[1, 1], 3.0)
    testing.assert_equal(mat[1, 2], 4.0)
    testing.assert_equal(mat[0, 0], 0.0)


def test_assign_whole_view_shape_mismatch_raises() raises:
    """A source that does not match the view's shape is rejected."""
    var mat = la.matrix[DType.float64]([[0.0, 0.0], [0.0, 0.0]])
    var src = la.matrix[DType.float64]([[1.0, 2.0, 3.0]])
    with testing.assert_raises():
        assign(view_mut(mat, Slice(0, 2), Slice(0, 2)), src)


def test_to_matrix_materialises_strided_view() raises:
    """`to_matrix` turns a strided view into dense owned storage."""
    var mat = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )
    var strided = mat[0:2, 0:4:2]
    var dense = strided.to_matrix()
    testing.assert_equal(dense.nrows, 2)
    testing.assert_equal(dense.ncols, 2)
    testing.assert_true(dense.is_c_contiguous())
    testing.assert_equal(dense[0, 0], 1.0)
    testing.assert_equal(dense[0, 1], 3.0)
    testing.assert_equal(dense[1, 0], 5.0)
    testing.assert_equal(dense[1, 1], 7.0)


def test_to_matrix_is_independent_of_source() raises:
    """The materialised matrix owns its data and does not alias the view."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var dense = mat.view().to_matrix()
    mat[0, 0] = 99.0
    testing.assert_equal(dense[0, 0], 1.0)


def test_view_handle_copy_is_implicit() raises:
    """A view is an O(1) handle, so plain assignment copies the handle."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var v = view_mut(mat, Slice(0, 2), Slice(0, 2))
    var v2 = v
    v2[0, 0] = 42.0
    # Both handles see the same buffer.
    testing.assert_equal(v[0, 0], 42.0)
    testing.assert_equal(mat[0, 0], 42.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
