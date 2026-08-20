"""
Tests for the Phase 5.3 reductions and searches.

Three things are checked throughout, because each has its own way of going
wrong:

*Values*, against NumPy's answers computed by hand.

*Shapes*, because `axis` names the dimension that disappears and the two
indices run opposite ways --- `axis=0` collapses the rows and so returns
`1 x ncols`. An implementation that inverts them still produces plausible
numbers on a square matrix, so every axis test here uses a non-square one.

*Strided operands*, because every routine funnels through views. A routine
that works on `m` and not on `m[0:4:2, 1:5:2]` has hard-coded a stride
somewhere.
"""

import std.testing as testing
import linamo as la
from linamo.routines.functional import apply_along_axis, fold
from linamo.routines.logic import all, any
from linamo.routines.math import cumprod, max, min, prod
from linamo.routines.searching import argmax, argmin
from linamo.routines.sorting import argsort, sort, sort_inplace
from linamo.routines.statistics import cumsum, sum


def _m() raises -> la.Matrix[Float64]:
    """A 2x3 matrix: non-square, so a transposed axis cannot hide."""
    return la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# ===----------------------------------------------------------------------===#
# sum / cumsum
# ===----------------------------------------------------------------------===#


def test_sum_full() raises:
    testing.assert_equal(sum(_m()), 21.0)


def test_sum_axis0() raises:
    var s = sum(_m(), 0)
    testing.assert_equal(s.nrows(), 1)
    testing.assert_equal(s.ncols(), 3)
    testing.assert_equal(s[0, 0], 5.0)
    testing.assert_equal(s[0, 1], 7.0)
    testing.assert_equal(s[0, 2], 9.0)


def test_sum_axis1() raises:
    var s = sum(_m(), 1)
    testing.assert_equal(s.nrows(), 2)
    testing.assert_equal(s.ncols(), 1)
    testing.assert_equal(s[0, 0], 6.0)
    testing.assert_equal(s[1, 0], 15.0)


def test_sum_bad_axis_raises() raises:
    with testing.assert_raises():
        _ = sum(_m(), 2)


def test_sum_of_a_strided_view() raises:
    """A view with a step in both directions must reduce correctly."""
    var a = la.matrix[Float64](
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    # Rows 0 and 2, columns 1 and 3: 2, 4, 10, 12.
    testing.assert_equal(sum(a[0:4:2, 1:4:2]), 28.0)


def test_sum_of_a_transposed_matrix() raises:
    """Column-contiguous input takes the other fast path in `fold`."""
    var t = la.transpose(_m())
    testing.assert_equal(sum(t), 21.0)
    var s = sum(t, 0)
    testing.assert_equal(s.nrows(), 1)
    testing.assert_equal(s.ncols(), 2)
    testing.assert_equal(s[0, 0], 6.0)
    testing.assert_equal(s[0, 1], 15.0)


def test_cumsum_full() raises:
    var c = cumsum(_m())
    testing.assert_equal(c[0, 0], 1.0)
    testing.assert_equal(c[0, 2], 6.0)
    testing.assert_equal(c[1, 0], 10.0)
    testing.assert_equal(c[1, 2], 21.0)


def test_cumsum_axis0() raises:
    var c = cumsum(_m(), 0)
    testing.assert_equal(c[0, 1], 2.0)
    testing.assert_equal(c[1, 1], 7.0)


def test_cumsum_axis1() raises:
    var c = cumsum(_m(), 1)
    testing.assert_equal(c[1, 0], 4.0)
    testing.assert_equal(c[1, 2], 15.0)


# ===----------------------------------------------------------------------===#
# prod / cumprod / min / max
# ===----------------------------------------------------------------------===#


def test_prod_full() raises:
    testing.assert_equal(prod(_m()), 720.0)


def test_prod_axis0() raises:
    var p = prod(_m(), 0)
    testing.assert_equal(p[0, 0], 4.0)
    testing.assert_equal(p[0, 2], 18.0)


def test_cumprod_axis1() raises:
    var c = cumprod(_m(), 1)
    testing.assert_equal(c[0, 2], 6.0)
    testing.assert_equal(c[1, 2], 120.0)


def test_min_max_full() raises:
    testing.assert_equal(min(_m()), 1.0)
    testing.assert_equal(max(_m()), 6.0)


def test_min_max_axis() raises:
    var lo = min(_m(), 0)
    testing.assert_equal(lo.nrows(), 1)
    testing.assert_equal(lo[0, 0], 1.0)
    var hi = max(_m(), 1)
    testing.assert_equal(hi.nrows(), 2)
    testing.assert_equal(hi[0, 0], 3.0)
    testing.assert_equal(hi[1, 0], 6.0)


def test_min_of_an_empty_matrix_raises() raises:
    var e = la.zeros[Float64](0, 0)
    with testing.assert_raises():
        _ = min(e)


def test_min_ignores_the_seed() raises:
    """The accumulator is seeded with the first element, not a sentinel.

    A matrix whose smallest element is also its first would pass even if the
    seed leaked into the result, so this one puts the minimum last.
    """
    var a = la.matrix[Float64]([[9.0, 8.0], [7.0, -2.0]])
    testing.assert_equal(min(a), -2.0)
    testing.assert_equal(max(a), 9.0)


# ===----------------------------------------------------------------------===#
# argmin / argmax
# ===----------------------------------------------------------------------===#


def test_argmin_argmax_full() raises:
    var a = la.matrix[Float64]([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    testing.assert_equal(argmin(a), 3)
    testing.assert_equal(argmax(a), 4)


def test_argmin_axis0() raises:
    var a = la.matrix[Float64]([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    var r = argmin(a, 0)
    testing.assert_equal(r.nrows(), 1)
    testing.assert_equal(r.ncols(), 3)
    testing.assert_equal(r[0, 0], 1)
    testing.assert_equal(r[0, 1], 0)
    testing.assert_equal(r[0, 2], 0)


def test_argmax_axis1() raises:
    var a = la.matrix[Float64]([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    var r = argmax(a, 1)
    testing.assert_equal(r.nrows(), 2)
    testing.assert_equal(r.ncols(), 1)
    testing.assert_equal(r[0, 0], 0)
    testing.assert_equal(r[1, 0], 1)


def test_argmin_ties_take_the_first() raises:
    """NumPy returns the earliest extremum; so do we."""
    var a = la.matrix[Float64]([[2.0, 1.0, 1.0]])
    testing.assert_equal(argmin(a), 1)


# ===----------------------------------------------------------------------===#
# all / any
# ===----------------------------------------------------------------------===#


def test_all_any_numeric() raises:
    var a = la.matrix[Float64]([[3.0, 1.0], [0.0, 5.0]])
    testing.assert_false(all(a))
    testing.assert_true(any(a))


def test_all_any_all_zero() raises:
    var z = la.zeros[Float64](2, 2)
    testing.assert_false(all(z))
    testing.assert_false(any(z))


def test_all_axis0() raises:
    var a = la.matrix[Float64]([[3.0, 1.0], [0.0, 5.0]])
    var r = all(a, 0)
    testing.assert_equal(r[0, 0], False)
    testing.assert_equal(r[0, 1], True)


def test_any_on_a_comparison_mask() raises:
    """The common case: `all`/`any` over what a comparison produced."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[1.0, 9.0], [3.0, 9.0]])
    var mask = a == b
    testing.assert_true(any(mask))
    testing.assert_false(all(mask))


# ===----------------------------------------------------------------------===#
# sort / argsort / sort_inplace
# ===----------------------------------------------------------------------===#


def test_sort_rows() raises:
    var a = la.matrix[Float64]([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    var s = sort(a, 1)
    testing.assert_equal(s[0, 0], 1.0)
    testing.assert_equal(s[0, 2], 3.0)
    testing.assert_equal(s[1, 0], 0.0)
    testing.assert_equal(s[1, 2], 5.0)


def test_sort_columns() raises:
    var a = la.matrix[Float64]([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    var s = sort(a, 0)
    testing.assert_equal(s[0, 0], 0.0)
    testing.assert_equal(s[1, 0], 3.0)
    testing.assert_equal(s[0, 1], 1.0)
    testing.assert_equal(s[1, 1], 5.0)


def test_sort_leaves_the_operand_alone() raises:
    var a = la.matrix[Float64]([[3.0, 1.0]])
    _ = sort(a, 1)
    testing.assert_equal(a[0, 0], 3.0)


def test_sort_inplace_rewrites_the_matrix() raises:
    var a = la.matrix[Float64]([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    sort_inplace(a, 1)
    testing.assert_equal(a[0, 0], 1.0)
    testing.assert_equal(a[0, 2], 3.0)
    testing.assert_equal(a[1, 0], 0.0)


def test_sort_inplace_keeps_the_layout() raises:
    """A column-major matrix must not be silently re-laid-out."""
    var a = la.matrix[Float64]([[3.0, 1.0], [0.0, 5.0]], order="F")
    var row_stride = a.row_stride()
    var col_stride = a.col_stride()
    sort_inplace(a, 1)
    testing.assert_equal(a.row_stride(), row_stride)
    testing.assert_equal(a.col_stride(), col_stride)
    testing.assert_equal(a[0, 0], 1.0)
    testing.assert_equal(a[0, 1], 3.0)


def test_argsort_matches_sort() raises:
    var a = la.matrix[Float64]([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    var order = argsort(a, 1)
    var sorted = sort(a, 1)
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            testing.assert_equal(sorted[i, j], a[i, Int(order[i, j])])


def test_argsort_is_stable() raises:
    """Equal elements keep their original relative order."""
    var a = la.matrix[Float64]([[1.0, 0.0, 1.0, 0.0]])
    var order = argsort(a, 1)
    testing.assert_equal(order[0, 0], 1)
    testing.assert_equal(order[0, 1], 3)
    testing.assert_equal(order[0, 2], 0)
    testing.assert_equal(order[0, 3], 2)


def test_sort_bad_axis_raises() raises:
    with testing.assert_raises():
        _ = sort(_m(), 7)


# ===----------------------------------------------------------------------===#
# The generic applier
# ===----------------------------------------------------------------------===#


def _count_positive[
    dtype: DType, origin: Origin[mut=False]
](v: la.MatrixView[Scalar[dtype], origin]) -> Scalar[dtype]:
    """A lane kernel that is not a fold, to show the applier is general."""
    var n = Scalar[dtype](0)
    for i in range(v.nrows()):
        for j in range(v.ncols()):
            if v[i, j] > 0:
                n += 1
    return n


def test_apply_along_axis_with_a_custom_kernel() raises:
    var a = la.matrix[Float64]([[1.0, -1.0, 2.0], [-3.0, -4.0, 5.0]])
    var r = apply_along_axis[
        axis=1, func=_count_positive[DType.float64, type_of(a.view()).origin]
    ](a.view())
    testing.assert_equal(r.nrows(), 2)
    testing.assert_equal(r.ncols(), 1)
    testing.assert_equal(r[0, 0], 2.0)
    testing.assert_equal(r[1, 0], 1.0)


def main() raises:
    var suite = testing.TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
