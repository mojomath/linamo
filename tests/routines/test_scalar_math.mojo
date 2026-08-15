"""
Tests for scalar-matrix operations: scalar_add, scalar_sub, scalar_mul, scalar_div.
"""

import std.testing as testing
import linamo as la
from linamo.routines.math import scalar_add, scalar_sub, scalar_mul, scalar_div


def test_scalar_add() raises:
    """Test adding a scalar to every element."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var result = scalar_add(mat, 10.0)
    testing.assert_equal(result[0, 0], 11.0)
    testing.assert_equal(result[0, 1], 12.0)
    testing.assert_equal(result[1, 0], 13.0)
    testing.assert_equal(result[1, 1], 14.0)


def test_scalar_add_negative() raises:
    """Test adding a negative scalar (effectively subtraction)."""
    var mat = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var result = scalar_add(mat, -5.0)
    testing.assert_equal(result[0, 0], 5.0)
    testing.assert_equal(result[1, 1], 35.0)


def test_scalar_sub() raises:
    """Test subtracting a scalar from every element."""
    var mat = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var result = scalar_sub(mat, 5.0)
    testing.assert_equal(result[0, 0], 5.0)
    testing.assert_equal(result[0, 1], 15.0)
    testing.assert_equal(result[1, 0], 25.0)
    testing.assert_equal(result[1, 1], 35.0)


def test_scalar_mul() raises:
    """Test multiplying every element by a scalar."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var result = scalar_mul(mat, 3.0)
    testing.assert_equal(result[0, 0], 3.0)
    testing.assert_equal(result[0, 1], 6.0)
    testing.assert_equal(result[1, 0], 9.0)
    testing.assert_equal(result[1, 1], 12.0)


def test_scalar_mul_zero() raises:
    """Test multiplying by zero."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var result = scalar_mul(mat, 0.0)
    testing.assert_equal(result[0, 0], 0.0)
    testing.assert_equal(result[1, 1], 0.0)


def test_scalar_div() raises:
    """Test dividing every element by a scalar."""
    var mat = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])
    var result = scalar_div(mat, 10.0)
    testing.assert_equal(result[0, 0], 1.0)
    testing.assert_equal(result[0, 1], 2.0)
    testing.assert_equal(result[1, 0], 3.0)
    testing.assert_equal(result[1, 1], 4.0)


def test_scalar_ops_col_major() raises:
    """Test scalar ops on a column-major matrix."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]], order="F")
    var result = scalar_mul(mat, 2.0)
    testing.assert_equal(result[0, 0], 2.0)
    testing.assert_equal(result[0, 1], 4.0)
    testing.assert_equal(result[1, 0], 6.0)
    testing.assert_equal(result[1, 1], 8.0)


def test_scalar_ops_preserve_shape() raises:
    """Test that scalar ops preserve matrix shape."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = scalar_add(mat, 1.0)
    testing.assert_equal(result.nrows, 2)
    testing.assert_equal(result.ncols, 3)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
