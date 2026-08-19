"""
Tests for combining two views of the *same* matrix in one expression.

Every other view test pairs views taken from two different matrices, which
never exercises the borrow checker. These tests all read one matrix twice in
a single call, which is only legal because slicing yields a read-only view.
"""

import std.testing as testing
import linamo as la
from linamo.routines.math import add, sub, matmul
from linamo.routines.mutation import fill, rows_mut, store, view_mut


# ===----------------------------------------------------------------------===#
# Two slices of one matrix in one expression
# ===----------------------------------------------------------------------===#


def test_two_slices_of_one_matrix() raises:
    """Subtracting one row of a matrix from another must compile and run."""
    var a = la.matrix[DType.float64]([[10.0, 20.0], [1.0, 2.0]])
    var d = a[0:1, 0:2] - a[1:2, 0:2]
    testing.assert_equal(d[0, 0], 9.0)
    testing.assert_equal(d[0, 1], 18.0)


def test_matrix_plus_own_slice() raises:
    """A matrix and a view of itself are both readable at once."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = a + a[0:2, 0:2]
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 8.0)


def test_own_slice_plus_matrix() raises:
    """The reversed operand order works the same way."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = a[0:2, 0:2] + a
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 8.0)


def test_matmul_of_two_own_slices() raises:
    """Blocked algorithms multiply two blocks of one matrix."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = a[0:2, 0:2] @ a[0:2, 0:2]
    testing.assert_equal(c[0, 0], 7.0)
    testing.assert_equal(c[0, 1], 10.0)
    testing.assert_equal(c[1, 0], 15.0)
    testing.assert_equal(c[1, 1], 22.0)


def test_compare_two_own_slices() raises:
    """Comparison masks are subject to the same borrow rules."""
    var a = la.matrix[DType.float64]([[1.0, 5.0], [9.0, 2.0]])
    var mask = a[0:1, 0:2] < a[1:2, 0:2]
    testing.assert_equal(mask[0, 0], True)
    testing.assert_equal(mask[0, 1], False)


def test_strided_slices_of_one_matrix() raises:
    """Non-contiguous views of one matrix combine too."""
    var a = la.matrix[DType.float64](
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    # [[1, 3], [9, 11]] + [[6, 8], [14, 16]]
    var c = a[0:4:2, 0:4:2] + a[1:4:2, 1:4:2]
    testing.assert_equal(c[0, 0], 7.0)
    testing.assert_equal(c[0, 1], 11.0)
    testing.assert_equal(c[1, 0], 23.0)
    testing.assert_equal(c[1, 1], 27.0)


def test_view_on_view_of_one_parent() raises:
    """Two sub-views carved from one parent view also compose."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var parent = a[0:2, 0:2]
    var c = parent[0:1, 0:2] + parent[1:2, 0:2]
    testing.assert_equal(c[0, 0], 4.0)
    testing.assert_equal(c[0, 1], 6.0)


def test_named_routines_accept_two_own_slices() raises:
    """The free functions behave like the operators."""
    var a = la.matrix[DType.float64]([[10.0, 20.0], [1.0, 2.0]])
    var s = add(a[0:1, 0:2], a[1:2, 0:2])
    testing.assert_equal(s[0, 0], 11.0)
    var d = sub(a, a[0:2, 0:2])
    testing.assert_equal(d[1, 1], 0.0)
    var p = matmul(a[0:2, 0:2], a[0:2, 0:2])
    testing.assert_equal(p[0, 0], 120.0)


# ===----------------------------------------------------------------------===#
# Demoting a mutable view with `as_imm`
# ===----------------------------------------------------------------------===#


def test_as_imm_allows_a_mutable_view_to_be_reused() raises:
    """`view()` inherits mutability, so it needs `as_imm()` to be paired."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var v = a.view()
    var c = v.as_imm() + v.as_imm()
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 8.0)


def test_as_imm_preserves_shape_and_strides() raises:
    """Demoting the origin changes nothing else about the view."""
    var a = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var v = view_mut(a, Slice(0, 3, 2), Slice(0, 3, 2))
    var i = v.as_imm()
    testing.assert_equal(i.nrows(), v.nrows())
    testing.assert_equal(i.ncols(), v.ncols())
    testing.assert_equal(i.row_stride(), v.row_stride())
    testing.assert_equal(i.col_stride(), v.col_stride())
    testing.assert_equal(i.offset(), v.offset())
    testing.assert_equal(i[1, 1], 9.0)


# ===----------------------------------------------------------------------===#
# The mutable door: `view_mut(m, x, y)`
# ===----------------------------------------------------------------------===#


def test_view_mut_is_writable() raises:
    """`view_mut(m, x, y)` inherits mutability where `m[x, y]` does not."""
    var a = la.matrix[DType.float64](
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    fill(
        view_mut(a, Slice(0, 2), Slice(1, 3)),
        Slice(0, 2),
        Slice(0, 2),
        7.0,
    )
    testing.assert_equal(a[0, 1], 7.0)
    testing.assert_equal(a[1, 2], 7.0)
    testing.assert_equal(a[0, 0], 0.0)
    testing.assert_equal(a[2, 2], 0.0)


def test_view_mut_is_strided() raises:
    """A strided mutable sub-view writes to the right elements."""
    var a = la.matrix[DType.float64](
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    var strided = view_mut(a, Slice(0, 2), Slice(0, 4, 2))
    store[width=2](strided, 1, 0, SIMD[DType.float64, 2](5.0, 6.0))
    testing.assert_equal(a[1, 0], 5.0)
    testing.assert_equal(a[1, 2], 6.0)
    testing.assert_equal(a[1, 1], 0.0)


def test_nested_mutable_sub_view() raises:
    """`view_mut` on a writable view keeps the parent view's mutability."""
    var a = la.matrix[DType.float64](
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    var outer = view_mut(a, Slice(0, 3), Slice(0, 3))
    var inner = view_mut(outer, Slice(1, 3), Slice(1, 3))
    fill(inner, Slice(0, 2), Slice(0, 2), 3.0)
    testing.assert_equal(a[1, 1], 3.0)
    testing.assert_equal(a[2, 2], 3.0)
    testing.assert_equal(a[0, 0], 0.0)


# ===----------------------------------------------------------------------===#
# Reading one matrix twice without any views involved
# ===----------------------------------------------------------------------===#


def test_two_elements_of_one_matrix() raises:
    """Adding two elements of one matrix must compile.

    `Matrix.__getitem__` used to return a reference whose origin named a single
    computed element, and forming a second such reference invalidated the
    first, so this expression did not compile on a `var` matrix.
    """
    var a = la.matrix[DType.float64]([[10.0, 20.0], [1.0, 2.0]])
    testing.assert_equal(a[0, 0] + a[1, 1], 12.0)
    testing.assert_equal(a[0, 0] + a[0, 1] + a[1, 0] + a[1, 1], 33.0)


def test_element_write_reading_the_same_matrix() raises:
    """A write may read the same matrix on the right-hand side."""
    var a = la.matrix[DType.float64]([[10.0, 20.0], [1.0, 2.0]])
    a[0, 1] = a[1, 0] + 1.0
    testing.assert_equal(a[0, 1], 2.0)
    a[0, 0] += 5.0
    testing.assert_equal(a[0, 0], 15.0)


def test_whole_matrix_view_is_read_only() raises:
    """`view()` is a read-only conversion, so it composes with itself.

    It used to take `ref self`, which made `m.view()` a silent write door and
    made `v + v` two mutable borrows of one matrix.
    """
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var v = a.view()
    var c = v + v
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 8.0)


def test_view_equals_full_slice() raises:
    """`m.view()` is a shorthand for `m[:, :]`."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var full = a[0:2, 0:2]
    var v = a.view()
    testing.assert_equal(v.nrows(), full.nrows())
    testing.assert_equal(v.ncols(), full.ncols())
    testing.assert_equal(v[1, 1], full[1, 1])


# ===----------------------------------------------------------------------===#
# Writable iteration is opt-in
# ===----------------------------------------------------------------------===#


def test_rows_mut_writes_through() raises:
    """`rows_mut` yields writable rows; plain `rows()` does not."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    for row in rows_mut(a):
        row[0, 0] = 99.0
    testing.assert_equal(a[0, 0], 99.0)
    testing.assert_equal(a[1, 0], 99.0)
    testing.assert_equal(a[0, 1], 2.0)


def test_plain_iteration_still_reads() raises:
    """Read-only iteration keeps working and stays zero-copy."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var total = Float64(0.0)
    for row in a:
        total += row[0, 0] + row[0, 1]
    testing.assert_equal(total, 10.0)


def test_cols_mut_writes_through() raises:
    """`cols_mut` walks columns writably."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    from linamo.routines.mutation import cols_mut

    for col in cols_mut(a):
        col[0, 0] = 7.0
    testing.assert_equal(a[0, 0], 7.0)
    testing.assert_equal(a[0, 1], 7.0)
    testing.assert_equal(a[1, 0], 3.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
