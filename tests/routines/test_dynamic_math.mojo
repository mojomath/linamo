"""
Tests for dynamic Matrix element-wise operations: add, sub, mul, div.
"""

import std.testing as testing
import linamo as la
from linamo.routines.math import add, sub, mul, div


# ===----------------------------------------------------------------------===#
# Dynamic add
# ===----------------------------------------------------------------------===#


def test_dynamic_add() raises:
    """Test element-wise addition of two dynamic matrices."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    var c = add(a, b)
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[0, 1], 22.0)
    testing.assert_equal(c[1, 0], 33.0)
    testing.assert_equal(c[1, 1], 44.0)


def test_dynamic_add_dunder() raises:
    """Test element-wise addition using + operator."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[5.0, 6.0], [7.0, 8.0]])
    var c = a + b
    testing.assert_equal(c[0, 0], 6.0)
    testing.assert_equal(c[1, 1], 12.0)


def test_dynamic_add_shape_mismatch_raises() raises:
    """Test that adding matrices of different shapes raises ValueError."""
    var a = la.matrix[Float64]([[1.0, 2.0]])
    var b = la.matrix[Float64]([[1.0], [2.0]])
    var raised = False
    try:
        var _c = add(a, b)
    except:
        raised = True
    testing.assert_true(raised, "Shape mismatch should raise ValueError")


# ===----------------------------------------------------------------------===#
# Dynamic sub
# ===----------------------------------------------------------------------===#


def test_dynamic_sub() raises:
    """Test element-wise subtraction of two dynamic matrices."""
    var a = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    var b = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = sub(a, b)
    testing.assert_equal(c[0, 0], 9.0)
    testing.assert_equal(c[0, 1], 18.0)
    testing.assert_equal(c[1, 0], 27.0)
    testing.assert_equal(c[1, 1], 36.0)


def test_dynamic_sub_dunder() raises:
    """Test element-wise subtraction using - operator."""
    var a = la.matrix[Float64]([[5.0, 6.0], [7.0, 8.0]])
    var b = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = a - b
    testing.assert_equal(c[0, 0], 4.0)
    testing.assert_equal(c[1, 1], 4.0)


# ===----------------------------------------------------------------------===#
# Dynamic mul
# ===----------------------------------------------------------------------===#


def test_dynamic_mul() raises:
    """Test element-wise multiplication of two dynamic matrices."""
    var a = la.matrix[Float64]([[2.0, 3.0], [4.0, 5.0]])
    var b = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    var c = mul(a, b)
    testing.assert_equal(c[0, 0], 20.0)
    testing.assert_equal(c[0, 1], 60.0)
    testing.assert_equal(c[1, 0], 120.0)
    testing.assert_equal(c[1, 1], 200.0)


def test_dynamic_mul_dunder() raises:
    """Test element-wise multiplication using * operator."""
    var a = la.matrix[Float64]([[2.0, 3.0], [4.0, 5.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [2.0, 2.0]])
    var c = a * b
    testing.assert_equal(c[0, 0], 4.0)
    testing.assert_equal(c[1, 1], 10.0)


# ===----------------------------------------------------------------------===#
# Dynamic div
# ===----------------------------------------------------------------------===#


def test_dynamic_div() raises:
    """Test element-wise division of two dynamic matrices."""
    var a = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    var b = la.matrix[Float64]([[2.0, 4.0], [5.0, 8.0]])
    var c = div(a, b)
    testing.assert_equal(c[0, 0], 5.0)
    testing.assert_equal(c[0, 1], 5.0)
    testing.assert_equal(c[1, 0], 6.0)
    testing.assert_equal(c[1, 1], 5.0)


def test_dynamic_div_dunder() raises:
    """Test element-wise division using / operator."""
    var a = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    var b = la.matrix[Float64]([[5.0, 5.0], [5.0, 5.0]])
    var c = a / b
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 8.0)


# ===----------------------------------------------------------------------===#
# Mixed layout operations
# ===----------------------------------------------------------------------===#


def test_dynamic_add_col_major() raises:
    """Test addition with column-major matrices."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]], order="F")
    var b = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]], order="F")
    var c = add(a, b)
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[0, 1], 22.0)
    testing.assert_equal(c[1, 0], 33.0)
    testing.assert_equal(c[1, 1], 44.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
