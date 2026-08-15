"""
Tests for view-based element-wise operations (view×view, mat×view, view×mat)
and view-based scalar operations.
"""

import std.testing as testing
import linamo as la
from linamo.routines.math import add, sub, mul, div
from linamo.routines.math import scalar_add, scalar_sub, scalar_mul, scalar_div


# ===----------------------------------------------------------------------===#
# Element-wise: view × view
# ===----------------------------------------------------------------------===#


def test_add_view_view() raises:
    """Test add on two views (sub-matrices)."""
    var a = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var b = la.matrix[DType.float64](
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]
    )
    # Take 2x2 sub-views from top-left
    var va = a[0:2, 0:2]
    var vb = b[0:2, 0:2]
    var c = add(va, vb)
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[0, 1], 22.0)
    testing.assert_equal(c[1, 0], 44.0)
    testing.assert_equal(c[1, 1], 55.0)


def test_sub_view_view() raises:
    """Test sub on two views."""
    var a = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var b = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = sub(a[0:2, 0:2], b[0:2, 0:2])
    testing.assert_equal(c[0, 0], 9.0)
    testing.assert_equal(c[1, 1], 36.0)


def test_mul_view_view() raises:
    """Test mul on two views."""
    var a = la.matrix[DType.float64]([[2.0, 3.0], [4.0, 5.0]])
    var b = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var c = mul(a[0:2, 0:2], b[0:2, 0:2])
    testing.assert_equal(c[0, 0], 20.0)
    testing.assert_equal(c[0, 1], 60.0)
    testing.assert_equal(c[1, 0], 120.0)
    testing.assert_equal(c[1, 1], 200.0)


def test_div_view_view() raises:
    """Test div on two views."""
    var a = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var b = la.matrix[DType.float64]([[2.0, 4.0], [5.0, 8.0]])
    var c = div(a[0:2, 0:2], b[0:2, 0:2])
    testing.assert_equal(c[0, 0], 5.0)
    testing.assert_equal(c[1, 1], 5.0)


# ===----------------------------------------------------------------------===#
# Element-wise: mat × view
# ===----------------------------------------------------------------------===#


def test_add_mat_view() raises:
    """Test add: Matrix + MatrixView."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var big = la.matrix[DType.float64](
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]
    )
    var vb = big[0:2, 0:2]  # [[10, 20], [40, 50]]
    var c = add(a, vb)
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[0, 1], 22.0)
    testing.assert_equal(c[1, 0], 43.0)
    testing.assert_equal(c[1, 1], 54.0)


def test_sub_mat_view() raises:
    """Test sub: Matrix - MatrixView."""
    var a = la.matrix[DType.float64]([[100.0, 200.0], [300.0, 400.0]])
    var big = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var vb = big[0:2, 0:2]
    var c = sub(a, vb)
    testing.assert_equal(c[0, 0], 99.0)
    testing.assert_equal(c[1, 1], 395.0)


# ===----------------------------------------------------------------------===#
# Element-wise: view × mat
# ===----------------------------------------------------------------------===#


def test_add_view_mat() raises:
    """Test add: MatrixView + Matrix."""
    var big = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var va = big[0:2, 0:2]  # [[1, 2], [4, 5]]
    var b = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var c = add(va, b)
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[0, 1], 22.0)
    testing.assert_equal(c[1, 0], 34.0)
    testing.assert_equal(c[1, 1], 45.0)


def test_mul_view_mat() raises:
    """Test mul: MatrixView * Matrix."""
    var big = la.matrix[DType.float64]([[2.0, 3.0, 99.0], [4.0, 5.0, 99.0]])
    var va = big[0:2, 0:2]  # [[2, 3], [4, 5]]
    var b = la.matrix[DType.float64]([[10.0, 10.0], [10.0, 10.0]])
    var c = mul(va, b)
    testing.assert_equal(c[0, 0], 20.0)
    testing.assert_equal(c[0, 1], 30.0)
    testing.assert_equal(c[1, 0], 40.0)
    testing.assert_equal(c[1, 1], 50.0)


# ===----------------------------------------------------------------------===#
# Strided views (non-contiguous)
# ===----------------------------------------------------------------------===#


def test_add_strided_views() raises:
    """Test add on views with step > 1 (non-contiguous)."""
    var a = la.matrix[DType.float64](
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    var b = la.matrix[DType.float64](
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
            [90.0, 100.0, 110.0, 120.0],
            [130.0, 140.0, 150.0, 160.0],
        ]
    )
    # Every other row and column: a[::2, ::2] = [[1, 3], [9, 11]]
    var va = a[0:4:2, 0:4:2]
    var vb = b[0:4:2, 0:4:2]  # [[10, 30], [90, 110]]
    var c = add(va, vb)
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[0, 1], 33.0)
    testing.assert_equal(c[1, 0], 99.0)
    testing.assert_equal(c[1, 1], 121.0)


# ===----------------------------------------------------------------------===#
# Scalar ops on views
# ===----------------------------------------------------------------------===#


def test_scalar_add_view() raises:
    """Test scalar_add on a MatrixView."""
    var big = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var v = big[0:2, 0:2]  # [[1, 2], [4, 5]]
    var c = scalar_add(v, 100.0)
    testing.assert_equal(c[0, 0], 101.0)
    testing.assert_equal(c[0, 1], 102.0)
    testing.assert_equal(c[1, 0], 104.0)
    testing.assert_equal(c[1, 1], 105.0)


def test_scalar_mul_view() raises:
    """Test scalar_mul on a MatrixView."""
    var big = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var v = big[0:2, 0:2]
    var c = scalar_mul(v, 10.0)
    testing.assert_equal(c[0, 0], 10.0)
    testing.assert_equal(c[0, 1], 20.0)
    testing.assert_equal(c[1, 0], 40.0)
    testing.assert_equal(c[1, 1], 50.0)


def test_scalar_sub_view() raises:
    """Test scalar_sub on a MatrixView."""
    var big = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var v = big[0:2, 0:2]
    var c = scalar_sub(v, 5.0)
    testing.assert_equal(c[0, 0], 5.0)
    testing.assert_equal(c[1, 1], 35.0)


def test_scalar_div_view() raises:
    """Test scalar_div on a MatrixView."""
    var big = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var v = big[0:2, 0:2]
    var c = scalar_div(v, 10.0)
    testing.assert_equal(c[0, 0], 1.0)
    testing.assert_equal(c[1, 1], 4.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
