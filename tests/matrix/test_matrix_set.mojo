"""
Tests for `Matrix.set`, the single spelling for every write, and for the
extent of a view built from a backwards or empty slice.
"""

import std.testing as testing
import linamo as la
from linamo.routines.mutation import fill, view_mut


# ===----------------------------------------------------------------------===#
# `set` dispatches on its arguments
# ===----------------------------------------------------------------------===#


def test_set_whole_matrix_scalar() raises:
    """`m.set(v)` writes one scalar into every element."""
    var m = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    m.set(7.0)
    for i in range(2):
        for j in range(2):
            testing.assert_equal(m[i, j], 7.0)


def test_set_whole_matrix_source() raises:
    """`m.set(src)` copies a whole matrix of the same shape."""
    var m = la.zeros[DType.float64](2, 3)
    var src = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m.set(src)
    testing.assert_equal(m[0, 0], 1.0)
    testing.assert_equal(m[1, 2], 6.0)


def test_set_whole_matrix_shape_mismatch_raises() raises:
    """A whole-matrix `set` still checks the shape."""
    var m = la.zeros[DType.float64](2, 3)
    var src = la.matrix[DType.float64]([[1.0, 2.0]])
    with testing.assert_raises():
        m.set(src)


def test_set_region_scalar() raises:
    """`m.set(rows, cols, v)` leaves everything outside the region alone."""
    var m = la.zeros[DType.float64](4, 4)
    m.set(Slice(1, 3), Slice(1, 3), 5.0)
    testing.assert_equal(m[1, 1], 5.0)
    testing.assert_equal(m[2, 2], 5.0)
    testing.assert_equal(m[0, 0], 0.0)
    testing.assert_equal(m[3, 3], 0.0)


def test_set_region_strided() raises:
    """A step in either slice selects every other row and column."""
    var m = la.zeros[DType.float64](4, 4)
    m.set(Slice(0, 4, 2), Slice(0, 4, 2), 1.0)
    testing.assert_equal(m[0, 0], 1.0)
    testing.assert_equal(m[0, 2], 1.0)
    testing.assert_equal(m[2, 0], 1.0)
    testing.assert_equal(m[2, 2], 1.0)
    testing.assert_equal(m[0, 1], 0.0)
    testing.assert_equal(m[1, 1], 0.0)


def test_set_region_from_matrix() raises:
    """A `Matrix` source is accepted directly, without `.view()`."""
    var m = la.zeros[DType.float64](4, 4)
    var block = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    m.set(Slice(2, 4), Slice(2, 4), block)
    testing.assert_equal(m[2, 2], 1.0)
    testing.assert_equal(m[3, 3], 4.0)
    testing.assert_equal(m[0, 0], 0.0)


def test_set_region_from_view() raises:
    """A sliced view is an equally good source."""
    var m = la.zeros[DType.float64](3, 3)
    var src = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    m.set(Slice(0, 2), Slice(0, 2), src[1:3, 1:3])
    testing.assert_equal(m[0, 0], 5.0)
    testing.assert_equal(m[1, 1], 9.0)


def test_set_region_from_mutable_view() raises:
    """A writable view may be read from as a source."""
    var m = la.zeros[DType.float64](2, 2)
    var src = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    m.set(Slice(0, 2), Slice(0, 2), view_mut(src, Slice(0, 2), Slice(0, 2)))
    testing.assert_equal(m[1, 0], 3.0)


def test_set_element() raises:
    """The element form agrees with subscript assignment."""
    var m = la.zeros[DType.float64](2, 2)
    m.set(0, 1, 3.0)
    testing.assert_equal(m[0, 1], 3.0)
    m[0, 1] = 4.0
    testing.assert_equal(m[0, 1], 4.0)


def test_set_element_out_of_bounds_raises() raises:
    """The element form is bounds-checked."""
    var m = la.zeros[DType.float64](2, 2)
    with testing.assert_raises():
        m.set(2, 0, 1.0)


def test_set_integer_dtype() raises:
    """Dispatch is by argument kind, not by dtype."""
    var m = la.zeros[DType.int64](2, 2)
    m.set(Slice(0, 1), Slice(0, 2), 3)
    testing.assert_equal(m[0, 0], 3)
    testing.assert_equal(m[1, 0], 0)


def test_set_region_writes_through_f_contiguous() raises:
    """An F-contiguous matrix is written through its own strides."""
    var m = la.zeros[DType.float64](3, 3)
    var f = la.reorder_layout(m)
    f.set(Slice(0, 2), Slice(0, 2), 2.0)
    testing.assert_equal(f[0, 0], 2.0)
    testing.assert_equal(f[1, 1], 2.0)
    testing.assert_equal(f[2, 2], 0.0)


# ===----------------------------------------------------------------------===#
# ===----------------------------------------------------------------------===#
# `view_mut` as a method
# ===----------------------------------------------------------------------===#


def whole_matrix_write_needs_no_try(mut m: la.Matrix[DType.float64]):
    """`set(value)` is callable from a non-raising context.

    This `def` carries no `raises`, so it would not compile if the whole-matrix
    scalar write could fail. That is the assertion; the body is incidental.
    """
    m.set(5.0)


def test_set_whole_matrix_does_not_raise() raises:
    """The whole-matrix scalar write is total, so it is declared non-raising."""
    var m = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    whole_matrix_write_needs_no_try(m)
    testing.assert_equal(m[0, 0], 5.0)
    testing.assert_equal(m[1, 1], 5.0)


def test_set_whole_matrix_covers_f_contiguous() raises:
    """The flat buffer walk covers every element under either layout."""
    var m = la.reorder_layout(
        la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    testing.assert_true(m.is_f_contiguous())
    m.set(0.0)
    for i in range(m.nrows()):
        for j in range(m.ncols()):
            testing.assert_equal(m[i, j], 0.0)


def test_view_mut_method_writes_through() raises:
    """`m.view_mut(...)` grants write access without importing the module."""
    var m = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var v = m.view_mut(Slice(0, 1), Slice(0, 2))
    v[0, 0] = 9.0
    v[0, 1] = 8.0
    testing.assert_equal(m[0, 0], 9.0)
    testing.assert_equal(m[0, 1], 8.0)
    testing.assert_equal(m[1, 0], 3.0)


def test_view_mut_method_matches_free_function() raises:
    """The method is a delegation, so both spellings select the same region."""
    var m = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var by_method = m.view_mut(Slice(1, 3), Slice(0, 3, 2))
    testing.assert_equal(by_method.nrows(), 2)
    testing.assert_equal(by_method.ncols(), 2)
    testing.assert_equal(by_method[0, 0], 4.0)
    testing.assert_equal(by_method[0, 1], 6.0)
    testing.assert_equal(by_method[1, 1], 9.0)


def test_view_mut_method_composes_with_mutation_routines() raises:
    """A view from the method satisfies the `Origin[mut=True]` routines."""
    var m = la.matrix[DType.float64]([[0.0, 0.0], [0.0, 0.0]])
    fill(
        m.view_mut(Slice(0, 2), Slice(0, 2)),
        Slice(0, 1),
        Slice(0, 2),
        Float64(7.0),
    )
    testing.assert_equal(m[0, 0], 7.0)
    testing.assert_equal(m[0, 1], 7.0)
    testing.assert_equal(m[1, 0], 0.0)


# Backwards and empty slices select nothing
# ===----------------------------------------------------------------------===#


def test_backwards_row_slice_is_empty() raises:
    """`m[3:1, :]` selects no rows, as in Python, rather than -2 of them."""
    var m = la.zeros[DType.float64](5, 6)
    var v = m[3:1, 0:6]
    testing.assert_equal(v.nrows(), 0)
    testing.assert_equal(v.ncols(), 6)
    testing.assert_equal(len(v), 0)


def test_backwards_col_slice_is_empty() raises:
    """The same holds in the column dimension."""
    var m = la.zeros[DType.float64](5, 6)
    var v = m[0:5, 4:1]
    testing.assert_equal(v.nrows(), 5)
    testing.assert_equal(v.ncols(), 0)


def test_negative_step_going_the_wrong_way_is_empty() raises:
    """`1:4:-1` walks away from its stop, so it selects nothing."""
    var m = la.zeros[DType.float64](5, 6)
    var v = m[1:4:-1, 0:6]
    testing.assert_equal(v.nrows(), 0)


def test_genuine_negative_step_still_counts_correctly() raises:
    """Clamping must not disturb a reversing slice that does select rows."""
    var m = la.matrix[DType.float64](
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )
    var v = m[4:0:-1, 0:2]
    testing.assert_equal(v.nrows(), 4)
    testing.assert_equal(v[0, 0], 9.0)
    testing.assert_equal(v[3, 0], 3.0)


def test_set_over_a_backwards_slice_writes_nothing() raises:
    """`set` inherits the empty selection, so the matrix is untouched."""
    var m = la.zeros[DType.float64](5, 6)
    m.set(Slice(3, 1), Slice(0, 6), 1.0)
    for i in range(5):
        for j in range(6):
            testing.assert_equal(m[i, j], 0.0)


def test_empty_slice_of_a_view_is_empty() raises:
    """Sub-slicing a view backwards is empty too, not negative."""
    var m = la.zeros[DType.float64](5, 6)
    var outer = m[0:5, 0:6]
    var inner = outer[3:1, 0:6]
    testing.assert_equal(inner.nrows(), 0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
